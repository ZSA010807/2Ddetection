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
def main():
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 数据变换
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    # 创建数据集
    train_dataset = DetectionDataset(
        images_dir='data/train0001/img1',
        labels_path='data/train0001/gt/gt.txt',
        transform=transform,
        img_size=416
    )
    
    val_dataset = DetectionDataset(
        images_dir='data/val0002/img1',
        labels_path='data/val0002/gt/gt.txt',
        transform=transform,
        img_size=416
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=4, 
        shuffle=True, 
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=4, 
        shuffle=False, 
        collate_fn=collate_fn
    )
    
    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(val_dataset)}")
    
    # 创建模型
    model = SimpleDetectionModel(num_classes=1)
    
    # 训练模型
    train_losses, val_losses = train_model(
        model, train_loader, val_loader, 
        num_epochs=50, device=device
    )
    
    # 绘制损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.savefig('training_loss.png')
    plt.show()
    
    # 测试推理
    print("\n进行推理测试...")
    test_image_path = 'data/val0002/img1/00000001.jpg'
    
    # 加载最佳模型
    model.load_state_dict(torch.load('best_detection_model.pth'))
    
    detections = detect_objects(model, test_image_path, device)
    print(f"检测到 {len(detections)} 个目标")
    
    # 可视化结果
    visualize_detection(test_image_path, detections)

if __name__ == "__main__":
    main()
