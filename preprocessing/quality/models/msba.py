"""
Multi-Scale Binned Activation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Optional, Tuple, Union
import math
from torchvision import transforms as T
import numpy as np
from torch.cuda.amp import autocast 
from preprocessing.quality.models.hopenet import *

class DeNormalize:
    def __init__(self, mean, std):
        self.mean = torch.tensor(mean).view(1, 1,-1)
        self.std = torch.tensor(std).view( 1, 1, -1)
        
    def __call__(self, tensor):
        # 输入是 w,h,c
        tensor = tensor * self.std + self.mean
        tensor = torch.clamp(tensor, 0, 1)  
        tensor = (tensor.numpy() * 255).astype(np.uint8)
        return tensor
    
def fourier_pattern(img, alpha,beta=0.4,amptith=1000):
    h,w ,_= img.shape
    if isinstance(alpha,torch.Tensor):
        alpha = torch.softmax(alpha,dim=0)[1].item() 
    alpha = float(alpha)
    alpha = 0 if alpha>0.5 else 1+alpha
    fft_source_cp = np.fft.fft2(img, axes=(0, 1))
    amp_source, pha_source = np.abs(fft_source_cp), np.angle(fft_source_cp)
    # np.fft.fftshift(amp_source, axes=(0, 1))
    amp_source_shift = amp_source  #

    b = (np.floor(np.amin((h, w)) * beta)).astype(int)  
    c_h = np.floor(h / 2.0).astype(int)
    c_w = np.floor(w / 2.0).astype(int)
    h1 = c_h - b
    h2 = c_h + b + 1
    w1 = c_w - b
    w2 = c_w + b + 1    
    # maxs= np.max(amp_source_shift[h1:h2, w1:w2,:])
    amp_source_shift[ h1:h2, w1:w2,:]  *= alpha 
    # amp_source_shift = np.fft.ifftshift(amp_source_shift, axes=(0, 1))
    fft_local_ = amp_source_shift * np.exp(1j * pha_source)
    local_in_trg = np.fft.ifft2(fft_local_, axes=(0, 1))
    local_in_trg = np.real(local_in_trg)
    local_in_trg=np.clip(local_in_trg, 0, 255).astype(np.uint8)
    return local_in_trg


class CVEmbeddings(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.embed_dim = 1024
        self.image_size = 224
        self.patch_size = 14

        self.class_embedding = nn.Parameter(torch.randn(self.embed_dim))

        self.patch_embedding = nn.Conv2d(
            in_channels=3,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=False,
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches + 1
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        self.register_buffer("position_ids", torch.arange(self.num_positions).expand((1, -1)), persistent=False)

    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        batch_size = pixel_values.shape[0]
        target_dtype = self.patch_embedding.weight.dtype
        patch_embeds = self.patch_embedding(pixel_values.to(dtype=target_dtype))  # shape = [*, width, grid, grid]
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)

        class_embeds = self.class_embedding.expand(batch_size, 1, -1)
        embeddings = torch.cat([class_embeds, patch_embeds], dim=1)
        embeddings = embeddings + self.position_embedding(self.position_ids)
        return embeddings



class QuickGELUActivation(nn.Module):
    def forward(self, input):
        return input * torch.sigmoid(1.702 * input)
class MLP(nn.Module):
    def __init__(self,):
        super().__init__()
        self.activation_fn = QuickGELUActivation()
        self.fc1 = nn.Linear(1024, 4096)
        self.fc2 = nn.Linear(4096, 1024)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states

class HUGncoderLayer(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.embed_dim = 1024
        self.attnc = MSBAAttention()
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=1e-5)
        self.mlp = MLP()
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=1e-5)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        causal_attention_mask: torch.Tensor,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor]:
        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, attn_weights = self.attnc(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs
class HugEncoder(nn.Module):

    def __init__(self,):
        super().__init__()
        self.layers = nn.ModuleList([HUGncoderLayer() for _ in range(24)])
        self.gradient_checkpointing = False

    def forward(
        self,
        inputs_embeds,
        attention_mask: Optional[torch.Tensor] = None,
        causal_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
       
        output_attentions = output_attentions if output_attentions is not None else False
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else False
        )
        return_dict = return_dict if return_dict is not None else False

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        hidden_states = inputs_embeds
        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            layer_outputs = encoder_layer(
                hidden_states,
                attention_mask,
                causal_attention_mask,
                output_attentions=output_attentions,
                )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

       
        return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)


class HOunet(nn.Module):
    def __init__(self,):
        super().__init__()
 
        embed_dim = 1024
        self.embeddings = CVEmbeddings()
        self.pre_layrnorm = nn.LayerNorm(embed_dim, eps=1e-5)
        self.encoder = HugEncoder()
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=1e-5)
        self.subface = nn.Linear(embed_dim, 2)
    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        output_attentions = output_attentions if output_attentions is not None else False
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else False
        )
        return_dict = return_dict if return_dict is not None else True


        hidden_states = self.embeddings(pixel_values)
        hidden_states = self.pre_layrnorm(hidden_states)

        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = encoder_outputs[0]
        pooled_output = last_hidden_state[:, 0, :]
        pooled_output = self.post_layernorm(pooled_output)

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return self.subface(pooled_output)


class MSBA(nn.Module):
    def __init__(self,path='', out_nc=3, bins=128):
        super().__init__()
        self.in_nc = out_nc * bins
        self.out_nc = out_nc
        self.bins = bins
        self.norm_factor = 2. / (self.bins - 1)
        self.facemsba = module_init(HOunet())
        if path:
            tmp=dict()
            for i in range(3):
                tmp.update(torch.load(path + f'quatily{i}.pth', 
                                weights_only=True))
            self.load_state_dict(tmp)
        if torch.cuda.is_available():
            self.cuda()
        if torch.cuda.is_bf16_supported():
            self.to(torch.bfloat16)
        self.device = next(self.facemsba.parameters()).device
        self.dtype = next(self.facemsba.parameters()).dtype
        self.means = [0.48145466, 0.4578275, 0.40821073]
        self.stds = [0.26862954, 0.26130258, 0.27577711]
        self.ransofer =  T.Compose([
            T.ToTensor(),
            T.Normalize(mean=self.means, std=self.stds)
        ])
        self.eval()
    def decode_bboxes(
        self,
        loc: torch.Tensor,
        priors: torch.Tensor,
    ) -> torch.Tensor:
        """Decodes bounding boxes from predictions.

        Takes the predicted bounding boxes (locations) and undoes the 
        encoding for offset regression used at training time.

        Args:
            loc: Bounding box (location) predictions for loc layers of
                shape (N, out_dim, 4). 
            priors: Prior boxes in center-offset form of shape
                (out_dim, 4).

        Returns:
            A tensor of shape (N, out_dim, 4) representing decoded
            bounding box predictions where the last dim can be
            interpreted as x1, y1, x2, y2 coordinates - the start and
            the end corners defining the face box.
        """
        # Concatenate priors
        boxes = torch.cat((
            priors[:, :2] + loc[..., :2] * self.variance[0] * priors[:, 2:],
            priors[:, 2:] * torch.exp(loc[..., 2:] * self.variance[1])
        ), 2)
        
        # Adjust values for proper xy coords
        boxes[..., :2] -= boxes[..., 2:] / 2
        boxes[..., 2:] += boxes[..., :2]

        return boxes
    @torch.no_grad()  
    def forward(self, x, infer=None):
        # assert x.shape[1] == self.in_nc
        # scales = self.scales.view(self.scales.shape + (1,) * (x.ndim - self.scales.ndim))
        if infer is None:
            tens = self.ransofer(x).unsqueeze(0).to(self.device,dtype=self.dtype)
            alpha = self.facemsba(tens)[0]
        else:
            alpha = infer
        return fourier_pattern(x,alpha)

def main():
    msba = MSBA()
    img = torch.rand(2, msba.in_nc, 64, 64)
    out = msba(img)
   
if __name__ == "__main__":
    main()
