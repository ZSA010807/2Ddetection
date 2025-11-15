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
def train_model(model, train_loader, val_loader, num_epochs=50, device='cuda'):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = DetectionLoss()
    
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        for batch_idx, (images, targets) in enumerate(train_loader):
            images = images.to(device)
            
            optimizer.zero_grad()
            predictions = model(images)
            loss, box_loss, obj_loss, cls_loss = criterion(predictions, targets)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.4f}')
        
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # 验证阶段
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(device)
                predictions = model(images)
                loss, _, _, _ = criterion(predictions, targets)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
        
        # 保存最佳模型
        if epoch == 0 or avg_val_loss < min(val_losses[:-1]):
            torch.save(model.state_dict(), 'best_detection_model.pth')
    
    return train_losses, val_losses
