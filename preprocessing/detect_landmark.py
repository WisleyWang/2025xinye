import os
import os.path as osp
import dlib
import cv2
import numpy as np
from imutils import face_utils
from multiprocessing import Pool, cpu_count
from threading import Thread
from multiprocessing import Queue
import pickle
from tqdm.auto import tqdm
import glob
import argparse
class FaceLandmarkDetector:
    def __init__(self):
        self.face_detector = dlib.get_frontal_face_detector()
        predictor_path = osp.join('preprocessing/dlib_tools/shape_predictor_81_face_landmarks.dat')
        self.face_predictor = dlib.shape_predictor(predictor_path)
    
    def detect_landmarks(self, image_path):
        """检测单张图片的landmark"""
        try:
            # 读取图片
            image = cv2.imread(image_path)
            if image is None:
                print(f"无法读取图片: {image_path}")
                return None
            
            # 转换为rgb
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            height, width = rgb.shape[:2]
            
            # 检测人脸
            faces = self.face_detector(rgb, 1)
            if len(faces) == 0:
                if height!=250 or width!=250:
                    # print(f"未检测到人脸: {image_path}")
                    height, width = rgb.shape[:2]
                    full_face_rect = dlib.rectangle(50, 50, width-100, height-100)
                    faces = [full_face_rect]
                 # 获取landmark
                    return None
                else: # 如果没有检测到人脸，且对其过的 则使用整个图像作为人脸区域
                    print(f"使用全脸: {image_path}")
                    ## 抠图
                    full_face_rect = dlib.rectangle(0, 0, width, height)
                    faces = [full_face_rect]
            # 获取landmark
            landmark = None
            if len(faces):
                 # For now only take the biggest face
                face = max(faces, key=lambda rect: rect.width() * rect.height())
                shape = self.face_predictor(rgb, face)
                landmark = face_utils.shape_to_np(shape)

            
            return landmark
        except Exception as e:
            print(f"处理图片 {image_path} 时出错: {str(e)}")
            return None

def worker_process(image_queue, process_id):
    """工作进程函数"""
    print(f"进程 {process_id} 启动")
    detector = FaceLandmarkDetector()
    
    while True:
        item = image_queue.get()
        if item is None:  # 结束信号
            break
        
        image_path, save_path = item
        if os.path.exists(save_path):
            continue
        landmark = detector.detect_landmarks(image_path)
        
        if landmark is not None:
            # 保存landmark到文件
            # with open(save_path, 'wb') as f:
            np.save(save_path, landmark)
        else:
            print('未检测到人脸 ',image_path)
    print(f"进程 {process_id} 退出")


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess images")
    parser.add_argument("--image_path", type=str, required=False, help="Path to the image", default="")
    parser.add_argument("--save_path", type=str, required=False, help="Path to the save root", default="")
    args = parser.parse_args()
    return args

def process_images_multi_processing(input_dir, output_dir, num_processes=None):
    """多进程处理图片"""
    if not osp.exists(output_dir):
        os.makedirs(output_dir)
    
    # 获取所有图片文件
    image_files = []
    image_files = glob.glob(f"{input_dir}/*/*")
    print('image nums:',len(image_files))
    if not image_files:
        print("未找到图片文件")
        return
    
    # 创建队列
    image_queue = Queue(maxsize=500)
    
    # 确定进程数
    if num_processes is None:
        num_processes = cpu_count()

    # 启动工作进程
    processes = []
    for i in range(num_processes):
        p = Process(target=worker_process, args=(image_queue, i))
        p.daemon = True
        p.start()
        
        processes.append(p)
    
    # 添加任务到队列
    for image_path in tqdm(image_files):
        # 构造输出路径
        rel_path = osp.relpath(image_path, input_dir)
        save_path = osp.join(output_dir, rel_path.split('.')[0] + '.npy')
        os.makedirs(osp.dirname(save_path), exist_ok=True)
        image_queue.put((image_path, save_path))
    
    # 添加结束信号
    for _ in range(num_processes):
        image_queue.put(None)
    
    # 等待所有进程完成
    for p in processes:
        p.join()    
    print("所有处理完成")

if __name__ == '__main__':
    from multiprocessing import Process
    args = parse_args()
    # 配置输入输出目录
    input_directory = args.image_path  # '/home/wisley/code/2025xinye_face/train_crop'
    output_directory = args.save_path #'/home/wisley/code/2025xinye_face/train_crop_landmarks'
    
    # 启动处理
    process_images_multi_processing(input_directory, output_directory, num_processes=4)