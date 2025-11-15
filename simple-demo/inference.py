import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision import transforms
import glob
def detect_objects(model, image_path, device='cuda', confidence_threshold=0.5):
    model.eval()
    
    # 加载和预处理图像
    image = Image.open(image_path).convert('RGB')
    original_size = image.size
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((416, 416))
    ])
    
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        predictions = model(input_tensor)
    
    # 解析预测结果（简化版本）
    pred_boxes = predictions[0, :, :, :, :4]  # 取第一个batch
    pred_obj = torch.sigmoid(predictions[0, :, :, :, 4:5])
    
    # 找到高置信度的检测
    detected_boxes = []
    
    for anchor_idx in range(3):
        for i in range(26):
            for j in range(26):
                if pred_obj[anchor_idx, i, j, 0] > confidence_threshold:
                    # 解码边界框
                    bx, by, bw, bh = pred_boxes[anchor_idx, i, j].cpu().numpy()
                    
                    # 转换为像素坐标
                    x_center = bx * original_size[0]
                    y_center = by * original_size[1]
                    width = bw * original_size[0]
                    height = bh * original_size[1]
                    
                    x1 = x_center - width/2
                    y1 = y_center - height/2
                    
                    detected_boxes.append({
                        'bbox': [x1, y1, width, height],
                        'confidence': pred_obj[anchor_idx, i, j, 0].item()
                    })
    
    return detected_boxes

def visualize_detection(image_path, detections):
    image = Image.open(image_path)
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(image)
    
    for detection in detections:
        x, y, w, h = detection['bbox']
        conf = detection['confidence']
        
        rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        ax.text(x, y-5, f'Conf: {conf:.2f}', bbox=dict(boxstyle="round,pad=0.3", fc='red', alpha=0.7))
    
    plt.axis('off')
    plt.tight_layout()
    plt.show()
