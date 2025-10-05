## 预处理
echo '预处理训练数据'
python3 preprocess_img_multiprocess.py --image_path ../data/data1/train/images   --save_path ../train_crop/xinye  --total_splits 8

echo '划分json'
python3 prepare_dataset_info.py --label_info ../data/data1/train.txt  --dataset_info_path ../dataset_json

## 训练
echo '训练第一个模型'
python3 training/train.py --task_target   facetransformer  --detector_path training/config/detector/orth_facevit.yaml

## 预处理
echo '准备landmark'
python3 preprocessing/detect_landmark.py --image_path  ../train_crop --save_path ../train_crop_landmarks

## 微调
echo 'sbi 微调 注意 config/detector/orth_facevit_sbi.yaml 中需手动配置待微调的pretrained  '
python3 training/train.py --task_target   facetransformer_sbi  --detector_path training/config/detector/orth_facevit_sbi.yaml


## 训练
echo '第二个模型'
python3 training/train.py --task_target  relpknet   --detector_path training/config/detector/replknet.yaml

## 微调同上
echo 'sbi 微调 注意配置微调权重 步骤同1'
# 可不做，可能有副作用