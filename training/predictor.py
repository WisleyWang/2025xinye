"""
eval pretained model.
"""
import argparse
import datetime
import json
import os

import os.path as osp
import pickle
import random
import time
import traceback
from collections import defaultdict
from copy import deepcopy
from os.path import join

import cv2
import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import yaml
from PIL import Image as pil_image
from tqdm import tqdm
import glob
from torchvision import transforms as T
from PIL import Image
# from dataset.abstract_dataset import DeepfakeAbstractBaseDataset
import json
from sklearn import metrics
from detectors import DETECTOR
from dataset.albu import DeNormalize
from torch.cuda.amp import autocast as autocast, GradScaler
import json
def parse_args():
    parser = argparse.ArgumentParser(description='predicter')
    # parser.add_argument('--input_csv', type=str, default="example/example_input1.csv")
    parser.add_argument('--img_dir', type=str, default="")
    parser.add_argument("--output_csv", type=str, default="example/output.csv")
    # parser.add_argument("--model_path", type=str, default="")
    # parser.add_argument("--config", type=str, default="configs/xception.yaml")
    parser.add_argument("--val", type=str, default='')
    args = parser.parse_args()
    return args

    


class BatchBaselineDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths:str, config):
        self.config = config
        self.image_paths = image_paths
        self.preprocessors = []
        self.face_extractors = []
        # for _ in range(self.workers):
        #     preprocessor, face_extractor = self.build_model()
        #     self.preprocessors.append(preprocessor)
        #     self.face_extractors.append(face_extractor)
        mean = config['mean']
        std = config['std']
        self.transfor = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=mean, std=std)
        ])
    def __len__(self):
        return len(self.image_paths)
    
    # def build_model(self,):
    #     config = yaml.safe_load(open("configs/xception.yaml"))
    #     config2 = yaml.safe_load(open("configs/test_config.yaml"))
    #     config.update(config2)

    #     face_extractor = AlignFacePreprocess(config)
    #     preprocess = ToTensor(config)
    #     return preprocess, face_extractor

    def preprocess_one_image(self, image_path:str, index:int):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        size = self.config['resolution']
        if image.shape[-1] !=size or image.shape[-2] != size:
            image = cv2.resize(image, (size, size),interpolation=cv2.INTER_CUBIC)
        
        # img = cv2.resize(image, (size, size), interpolation=cv2.INTER_CUBIC)
        image = Image.fromarray(np.array(image, dtype=np.uint8))
        input_tensor = self.transfor(image)
        # croped_face = self.face_extractors[index % self.workers](image)
        # if croped_face is None:
        #     croped_face = cv2.resize(image, (256, 256))
        # input_tensor = self.preprocessors[index % self.workers](croped_face)
        return input_tensor

    def __getitem__(self, index):
        return self.preprocess_one_image(self.image_paths[index], index), self.image_paths[index]

  


def predict_batch(args, config_path, model_path):
    config = yaml.safe_load(open(config_path))
    if model_path:
        # config['ckpt'] = model_path
        config['pretrained'] = model_path

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_class = DETECTOR[config['model_name']]
    predictor =  model_class(config).to(device)

    predictor.eval()
    # input_csv = args.input_csv
    
    if args.val:
        ## 这里手动加载 json
        print('local validation')
        # dfdc_test_sample
        val_dict = json.load(open(f"../dataset_json/{args.val}.json",'r'))
        real_path = [ os.path.join(args.img_dir,p) for p in val_dict['real']['frames']]
        fake_path =  [ os.path.join(args.img_dir,p) for p in val_dict['fake']['frames']]
        labels = [0]*len(real_path) + [1]*len(fake_path)
        image_paths = real_path + fake_path
    else:
        image_paths = sorted(glob.glob(os.path.join(args.img_dir,"*.jpg")))
    print("test nums:",len(image_paths))

    # save_path = './debug_test1'
    # os.makedirs(save_path, exist_ok=True)
    dataset = BatchBaselineDataset(image_paths, config)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=48, shuffle=False, num_workers=8)
    # denormalize = DeNormalize(config['mean'], config['std'])
    res = {}
    preds = []
    for images, image_ps in tqdm(dataloader):
        images = images.to(device)
        data_dict = {"image": images}
        with torch.no_grad():
            with autocast(dtype=torch.bfloat16):
                predictions1 = predictor(data_dict, inference=True) # {'cls': out_sha, 'feat': sha_feat,'prob': prob_sha}
            scores = predictions1['prob'].detach().cpu().numpy().tolist()
            preds.extend(scores)
        for ind, (image_path, score) in enumerate(zip(image_ps, scores)):
            res[osp.basename(image_path)] = score

            
    if args.val:
        fpr, tpr, thresholds = metrics.roc_curve(labels, preds)
        v_auc = metrics.auc(fpr, tpr)
        print("val auc:",v_auc)
        val_res = {}
        for ind, pp in enumerate(image_paths):
            val_res[pp] = (preds[ind], labels[ind])
        return  val_res
    return res
    
    



if __name__ == '__main__':
    args = parse_args()
    config_path = "training/config/detector/orth_facevit.yaml"
    model_path = "exps/final_model/facetransformer_ckpt_best.pth"

    
    res1 = predict_batch(args, config_path, model_path)

    config_path = "training/config/detector/replknet_sbi.yaml"
    model_path = "exps/final_model/replk_best_ckp.pth"
    res2 = predict_batch(args, config_path, model_path)


    if not args.val:
        res = []
        for name,score in res1.items():
            score = score*0.8 + res2[name] * 0.2
            res.append([name, score])
        output_csv = args.output_csv
        df_result = pd.DataFrame(res, columns=["image_name", "score"])
        df_result.to_csv(output_csv, index=False)
    else:
        labels = []
        scores = []
        for pp in res1.keys():
            sc1 = res1[pp][0]
            sc2 = res2[pp][0]
            labels.append(res1[pp][1])
            sc = sc1*0.8 + sc2 * 0.2
            scores.append(sc)
        fpr, tpr, thresholds = metrics.roc_curve(labels, scores)
        v_auc = metrics.auc(fpr, tpr)
        print("final val auc:",v_auc)

