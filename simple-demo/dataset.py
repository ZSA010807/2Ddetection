import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
import glob

class DetectionDataset(Dataset):
    def __init__(self, images_dir, labels_path, transform=None, img_size=416):
        self.images_dir = images_dir
        self.labels_path = labels_path
        self.transform = transform
        self.img_size = img_size
        self.image_files = sorted(glob.glob(os.path.join(images_dir, "*.jpg")))
        
        # 解析标签文件
        self.annotations = self._parse_annotations()
        
    def _parse_annotations(self):
        annotations = {}
        with open(self.labels_path, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) < 6:
                    continue
                
                frame = int(parts[0])
                bb_left = float(parts[2])
                bb_top = float(parts[3])
                bb_width = float(parts[4])
                bb_height = float(parts[5])
                
                if frame not in annotations:
                    annotations[frame] = []
                
                annotations[frame].append({
                    'bbox': [bb_left, bb_top, bb_width, bb_height],
                    'class_id': 1  # 所有类别都是1
                })
        
        return annotations
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # 加载图像
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('RGB')
        original_size = image.size  # (width, height)
        
        # 获取对应的帧号（从文件名提取）
        frame_num = int(os.path.basename(img_path).split('.')[0])
        
        # 获取该帧的标注
        bboxes = []
        labels = []
        
        if frame_num in self.annotations:
            for ann in self.annotations[frame_num]:
                x, y, w, h = ann['bbox']
                # 转换为YOLO格式 (x_center, y_center, width, height) 并归一化
                x_center = (x + w/2) / original_size[0]
                y_center = (y + h/2) / original_size[1]
                width = w / original_size[0]
                height = h / original_size[1]
                
                bboxes.append([x_center, y_center, width, height])
                labels.append(ann['class_id'])
        
        # 转换为tensor
        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)
        
        # 调整图像大小
        image = transforms.Resize((self.img_size, self.img_size))(image)
        
        # 创建目标tensor
        if len(bboxes) > 0:
            targets = torch.zeros((len(bboxes), 5))
            targets[:, 1:] = torch.tensor(bboxes)  # 第一列是类别
            targets[:, 0] = torch.tensor(labels)
        else:
            targets = torch.zeros((0, 5))
        
        return image, targets

def collate_fn(batch):
    images = []
    targets = []
    
    for img, tgt in batch:
        images.append(img)
        targets.append(tgt)
    
    return torch.stack(images), targets
