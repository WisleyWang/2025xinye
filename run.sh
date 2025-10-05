
# 推理入口
## 预处理
echo '预处理'
python3 preprocess_img_multiprocess.py --image_path ../data/data2/test2/images  --save_path ../test2_crop  --total_splits 8

echo '预测'
## 推理
python3 training/predictor.py  --img_dir ../test2_crop --output_csv submit/submit.csv

echo 'output result to submit/submit.csv'