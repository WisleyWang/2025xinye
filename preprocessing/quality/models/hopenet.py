import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from torchvision.models.resnet import Bottleneck
from typing import Any, Optional, Tuple, Union



class MSBAAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self,):
        super().__init__()
        self.embed_dim = 1024
        self.num_heads = 16
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )
        self.scale = self.head_dim**-0.5
        self.dropout = 0.0

        self.kj = nn.Linear(self.embed_dim, self.embed_dim)
        self.vj = nn.Linear(self.embed_dim, self.embed_dim)
        self.qj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        # 将tensor按照指定的维度进行reshape，然后进行transpose操作，最后进行contiguous操作
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        causal_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        bsz, tgt_len, embed_dim = hidden_states.size()

        # get query proj
        query_states = self.qj(hidden_states) * self.scale
        key_states = self._shape(self.kj(hidden_states), -1, bsz)
        value_states = self._shape(self.vj(hidden_states), -1, bsz)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
                f" {attn_weights.size()}"
            )

        # apply the causal_attention_mask first
        if causal_attention_mask is not None:
            if causal_attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is"
                    f" {causal_attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + causal_attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if output_attentions:
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_probs, value_states)

        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, tgt_len, embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped

def module_init(model):
    for name, module in model.named_children():
        if isinstance(module,MSBAAttention):
            for sub_name, sub_module in module.named_modules():
                if isinstance(sub_module, nn.modules.linear.Linear):
                    parent_module = module
                    sub_module_names = sub_name.split('.')
                    for module_name in sub_module_names[:-1]:
                        parent_module = getattr(parent_module, module_name)
                    setattr(parent_module, sub_module_names[-1], split_linear(sub_module))
        else:
            module_init(module)
    return model

class Hopenet(nn.Module):
    """ Defines a head pose estimation network with 3 output layers: yaw, pitch and roll.
    `"Fine-Grained Head Pose Estimation Without Keypoints" <https://arxiv.org/pdf/1710.00925.pdf>`_.

    Predicts Euler angles by binning and regression.

    Args:
        block (nn.Module): Main convolution block
        layers (list of ints): Number of blocks per intermediate layer
        num_bins (int): Number of regression bins
    """
    def __init__(self, block=Bottleneck, layers=(3, 4, 6, 3), num_bins=66):
        self.inplanes = 64
        super(Hopenet, self).__init__()
        self.idx_tensor = None
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7)
        self.fc_yaw = nn.Linear(512 * block.expansion, num_bins)
        self.fc_pitch = nn.Linear(512 * block.expansion, num_bins)
        self.fc_roll = nn.Linear(512 * block.expansion, num_bins)

        # Vestigial layer from previous experiments
        self.fc_finetune = nn.Linear(512 * block.expansion + 3, 3)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        pred_yaw = self.fc_yaw(x)
        pred_pitch = self.fc_pitch(x)
        pred_roll = self.fc_roll(x)

        yaw_predicted = F.softmax(pred_yaw, dim=1)
        pitch_predicted = F.softmax(pred_pitch, dim=1)
        roll_predicted = F.softmax(pred_roll, dim=1)

        if self.idx_tensor is None:
            self.idx_tensor = torch.arange(0, 66, out=torch.FloatTensor()).to(x.device)

        # Get continuous predictions in degrees.
        yaw_predicted = torch.sum(yaw_predicted * self.idx_tensor, axis=1).unsqueeze(1) * 3 - 99
        pitch_predicted = torch.sum(pitch_predicted * self.idx_tensor, axis=1).unsqueeze(1) * 3 - 99
        roll_predicted = torch.sum(roll_predicted * self.idx_tensor, axis=1).unsqueeze(1) * 3 - 99

        return torch.cat((yaw_predicted, pitch_predicted, roll_predicted), axis=1)


class FACELinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, init_weight=None):
        super(FACELinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r1 = nn.Parameter(torch.Tensor(1), requires_grad=True)
        self.r2 = nn.Parameter(torch.Tensor(1024, 1), requires_grad=True)
        self.r3 = nn.Parameter(torch.Tensor(1, 1024), requires_grad=True)

        self.weight_main = nn.Parameter(torch.Tensor(out_features, in_features), requires_grad=False)
        if init_weight is not None:
            self.weight_main.data.copy_(init_weight)
        else:
            nn.init.kaiming_uniform_(self.weight_main, a=math.sqrt(5))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
            nn.init.zeros_(self.bias)
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        residual_weight = self.r2 @ torch.diag(self.r1) @ self.r3
        weight = self.weight_main + residual_weight
        return F.linear(x, weight, self.bias)

    
def split_linear(module):
    if isinstance(module, nn.modules.linear.Linear):
        in_features = module.in_features
        out_features = module.out_features
        bias = module.bias is not None
        new_module = FACELinear(in_features, out_features, bias=bias, init_weight=module.weight.data.clone())
        if bias and module.bias is not None:
            new_module.bias.data.copy_(module.bias.data)
        return new_module
    else:
        return module