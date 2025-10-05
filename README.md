> # 第十届信也杯 吉尔伽美什团队

# Install requirements
```bash
pip install -r requirements.txt
```

## 训练pipeline
拉取镜像开启容器时注意需增加共享内存 `--shm-size=3G`
### 数据预处理准备
裁剪并对齐人脸
```
python3 preprocess_img_multiprocess.py --image_path <训练图片路径>   --save_path <处理后人脸保存路径>  --total_splits 8
```
示例:
```
python3 preprocess_img_multiprocess.py --image_path ../train/images   --save_path ../train_crop/xinye  --total_splits 8
```


### 准备json文件
读取`train.txt`, 通过json划分训练和验证集合
```
python3 prepare_dataset_info.py --label_info <train.txt 路径>  --dataset_info_path <json 文件保存路径>
```
示例:
```
python3 prepare_dataset_info.py --label_info ../traon.txt  --dataset_info_path ../dataset_json
```
生成`xy_train.json`,`xy_val.json`，也可直接使用 `dataset_json`下的json文件，但要注意里面的路径多添加了一级`xinye`文件目录
因此配置config时应该为 `rgb_dir: ../train_crop` ,而图片保存路径为`../train_crop/xinye`

### 准备config
配置 `training/config/train_config.yaml`
- 修改`rgb_dir`: <预处理后训练图片保存路径>
- 修改`dataset_json_folder`: <json文件所在文件夹路径>

配置训练(推理) config
- 配置`train_dataset`: [xy_train]
- 配置 `test_dataset`: [xy_val]

### 训练流程
1. 训练模型 1
预权重：vit_base_patch16_224.augreg_in1k
https://huggingface.co/timm/vit_base_patch16_224.augreg_in1k

```
python3 training/train.py --task_target   facetransformer  --detector_path training/config/detector/orth_facevit.yaml
```
模型权重保存在exp/下

2. sbi微调
基于训练权重，利用xy_val数据使用sbi微调模型(需要landmarks)
生成landmark
```
cd preprocessing
python3 detect_landmark.py --image_path <预处理后训练的图片保存路径>  --save_path <保存landmark的路径>
```
注意，sbi代码中加载landmak是替换img中路径命名的，代码如：`ld_p = img_path.replace('train_crop', 'train_crop_landmarks').split('.')[0]+ '.npy'`
因此，landmark保存路径需要为:`train_crop_landmarks`，预处理后训练图片保存在`train_crop`一致.
当然也可以自行修改`training/dataset/sbi_dataset.py` 中landmark路径的获取方式

3. 基于xy_val微调
调整config: orth_facevit.yaml
- 配置`train_dataset`: [xy_val]
- 配置`dataset_type`: 'sbi'
- 配置`pretrained`: <第一步的模型权重路径>
- `adam`下的学习了调节到`0.0001`，`nEpochs`设置为1

修改配置后微调命令：
```
python3 training/train.py --task_target   facetransformer  --detector_path training/config/detector/orth_facevit.yaml
```
详见train.sh 已进行配置，只需修改 `pretrained` 中的路径，建议使用第5 个epoch的权重，在 exp/xxx/train/epoch_4_ckpt/ckpt_best.pth

仅微调1个epoch，多了会有负作用

1. 训练模型 2
The ImageNet-1K pretrained weight files from： https://github.com/VisionRush/DeepFakeDefenders
RepLKNet: https://drive.google.com/file/d/1vo-P3XB6mRLUeDzmgv90dOu73uCeLfZN/view?usp=sharing
```
python3 training/train.py --task_target  relpknet   --detector_path training/config/detector/replknet.yaml
```



## 推理流程

### 预处理测试数据集(同训练)
```
python3 preprocess_img_multiprocess.py --image_path <待测试图片路径>   --save_path <处理后人脸保存路径>  --total_splits 8
```


### 推理
添加`training/inference.py`中相应配置和权重地址，模型融合
```
python3 training/inference.py  --img_dir <预处理后的测试集> --output_csv <输出结果文件路径>  
```