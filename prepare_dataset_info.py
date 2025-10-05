'''
description: 
param : 
return: 
Author: xinyebei@xinye.com
Date: 2025-04-23 14:37:05
LastEditors: xinyebei@xinye.com
'''
import argparse
import json
import os
import os.path as osp
from sklearn.model_selection import StratifiedKFold

def parse_args():
    parser = argparse.ArgumentParser(description='Prepare dataset info')
    parser.add_argument('--dataset_root_path', type=str, default='xinye', help='dataset root path')
    parser.add_argument('--label_info', type=str, default='./data/train.txt', help='label info')
    parser.add_argument('--mode', type=str, default='train', help='build dataset mode')
    parser.add_argument('--dataset_name', type=str, default='train', help='dataset name')

    parser.add_argument('--dataset_info_path', type=str, default='./data/dataset_info', help='dataset info path')

    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = parse_args()
    names, labels = [], []
    with open(args.label_info, 'r') as f:
        lines = f.readline()
        while lines:
            name, label = lines.strip().split(" ")   
            label = int(label)
            names.append(name)
            labels.append(label)
            lines = f.readline()

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for train_index, val_index in skf.split(names, labels):
        break
    dataset_dict = {"xy_train":{"real":{"frames":[],}, "fake":{"frames":[],}},
                    "xy_val":{"real":{"frames":[],}, "fake":{"frames":[],}}}
    for ind in train_index:
        label = labels[ind]
        name = osp.join(args.dataset_root_path,names[ind])
        if label == 0:
            dataset_dict["xy_train"]["real"]["frames"].append(name)
        else:
            dataset_dict["xy_train"]["fake"]["frames"].append(name)
    for ind in val_index:
        label = labels[ind]
        name = osp.join(args.dataset_root_path,names[ind])
        if label == 0:
            dataset_dict["xy_val"]["real"]["frames"].append(name)
        else:
            dataset_dict["xy_val"]["fake"]["frames"].append(name)
    print(f"train nums: fake: {len(dataset_dict['xy_train']['fake']['frames'])}, real: {len(dataset_dict['xy_train']['real']['frames'])}")
    print(f"val nums: fake: {len(dataset_dict['xy_val']['fake']['frames'])}, real: {len(dataset_dict['xy_val']['real']['frames'])}")
    if not osp.exists(args.dataset_info_path):
        os.makedirs(args.dataset_info_path, exist_ok=True) 
    for mode in ["xy_train", "xy_val"]:
        output_file_path = osp.join(args.dataset_info_path, f"{mode}.json")
        with open(output_file_path, 'w') as f:
            json.dump(dataset_dict[mode], f)

    # print the successfully generated dataset dictionary
    print(f"json generated successfully.")