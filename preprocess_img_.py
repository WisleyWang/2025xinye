import argparse
import concurrent.futures
import datetime
import glob
import logging
import os
import os.path as osp
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
# import dlib
import numpy as np
import yaml
import json
# from imutils import face_utils
from skimage import transform as trans
from tqdm import tqdm

from preprocessing.preprocessor import AlignFacePreprocess


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess images")
    parser.add_argument("--image_path", type=str, required=False, help="Path to the image", default="")
    parser.add_argument("--txt_info", default=None)
    parser.add_argument("--save_path", type=str, required=False, help="Path to the save root", default="")
    parser.add_argument("--total_splits", type=int, required=False, help="total splits in multiprocessing", default=4)
    parser.add_argument("--splits", type=int, required=False, help="selected splits", default=0)

    args = parser.parse_args()
    return args



if __name__ == "__main__":
    args = parse_args()
    image_path = args.image_path
    txt_info = args.txt_info

    params_list = []
    if txt_info:
        with open(txt_info, 'r') as f:
            lines = f.readline()
            while lines:
                name, bools = lines.strip().split(" ")   
                bools = int(bools)
                image_file = os.path.join(args.image_path,name)
                save_crop_name = osp.join(args.save_path, name)
                os.makedirs(os.path.dirname(save_crop_name), exist_ok=True)
                params_list.append((image_file, bools, save_crop_name))
                lines = f.readline()
    else:
        image_files = os.listdir(image_path)
        params_list = [(osp.join(image_path, f), None, osp.join(args.save_path, f)) for f in image_files  if f.endswith('.jpg') or f.endswith('.png') or f.endswith('.jpeg') ]
    params_list = sorted(params_list, key=lambda x: x[0])
    
    preprocessor = AlignFacePreprocess(None)
    

    def task_func(image_file, bools, save_crop_name,):
        if os.path.exists(save_crop_name):
            return
        # print(image_file,flush=True)
        image = cv2.imread(str(image_file)) 
        cropped_face = preprocessor(image,bools)

        if cropped_face is not None:
            cv2.imwrite(save_crop_name, cropped_face)
        else:
            print(f"[warning!!!!] preprocess failed for {image_file}, using origin image instead",flush=True)
            cv2.imwrite(save_crop_name, image)
    
    
    # params_list = []
    # for path in image_files:
    #     image_file = os.path.join(args.image_path,path)
    #     # print(image_file)
    #     save_crop_name = osp.join(save_crop_root, path)
    #     os.makedirs(os.path.dirname(save_crop_name), exist_ok=True)
    #     params_list.append((image_file, None, save_crop_name))
    
    splits = np.array_split(params_list, args.total_splits)
    params_list = splits[args.splits]
    
    
    with tqdm(total=len(params_list), desc="Processing") as pbar:
        for params in params_list:
            task_func(*params)
            pbar.update()

    
    
    
    

