'''
description: 
param : 
return: 
Author: xinyebei@xinye.com
Date: 2025-01-20 14:43:45
LastEditors: xinyebei@xinye.com
'''
import argparse
import multiprocessing
import torch.multiprocessing as mp
import os
import os.path as osp
import subprocess
 
def run_subprocess(commandStr):
    p = subprocess.Popen(commandStr, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=0, universal_newlines=True)

    while p.poll() is None:
        line = p.stdout.readline()
        line = line.strip('\n')
        if line:
            print(line)

    returnCode = p.returncode
    return returnCode





def parse_args():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--image_path', type=str, default='', help='')
    parser.add_argument('--txt_info', type=str, default='', help='')
    parser.add_argument('--save_path', type=str, default='', help='')
    parser.add_argument('--total_splits', type=int, default=4, help='')
    args = parser.parse_args()
    return args

def worker(name, idx, image_path,txt_path, save_path, total_splits):
    """子进程要执行的任务"""

    # command = f"export CUDA_VISIBLE_DEVICES={idx} && "
    command = ""
    command += "python ./preprocess_img_.py " + \
                f"--splits {idx} " + \
                f"--image_path {image_path} " + \
                f"--save_path {save_path} " + \
                f"--total_splits {total_splits} "
    if txt_path != "":
        command += f"--txt_info {txt_path} "
    run_subprocess(command)
    # os.system(command)

if __name__ == '__main__':

    mp.set_start_method('spawn', force=True)  
    args = parse_args()
    threads = args.total_splits
    
    processes = []

    for i in range(threads):
        p = mp.Process(target=worker, args=('test', i, args.image_path, args.txt_info,args.save_path, args.total_splits))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()


