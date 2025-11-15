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
class SimpleDetectionModel(nn.Module):
    def __init__(self, num_classes=1, img_size=416):
        super(SimpleDetectionModel, self).__init__()
        self.img_size = img_size
        self.num_classes = num_classes
        
        # 骨干网络 - 简单的CNN
        self.backbone = nn.Sequential(
            # 输入: 3 x 416 x 416
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 32 x 208 x 208
            
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 64 x 104 x 104
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 128 x 52 x 52
            
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 256 x 26 x 26
        )
        
        # 检测头 - 预测边界框和类别
        self.detection_head = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, (5 + num_classes) * 3, 1),  # 3个anchor, 每个预测5+num_classes个值
        )
        
        # 简单的anchor设置
        self.anchors = torch.tensor([[0.5, 0.5], [1.0, 1.0], [1.5, 1.5]])
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # 骨干网络特征提取
        features = self.backbone(x)  # [batch, 256, 26, 26]
        
        # 检测头
        predictions = self.detection_head(features)  # [batch, (5+num_classes)*3, 26, 26]
        
        # 重塑预测结果
        predictions = predictions.view(batch_size, 3, 5 + self.num_classes, 26, 26)
        predictions = predictions.permute(0, 1, 3, 4, 2)  # [batch, 3, 26, 26, 5+num_classes]
        
        return predictions
