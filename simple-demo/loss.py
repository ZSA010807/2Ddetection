class DetectionLoss(nn.Module):
    def __init__(self, num_classes=1):
        super(DetectionLoss, self).__init__()
        self.num_classes = num_classes
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
        
    def forward(self, predictions, targets):
        batch_size = predictions.size(0)
        
        # 解析预测结果
        pred_boxes = predictions[..., :4]  # [batch, 3, 26, 26, 4]
        pred_obj = predictions[..., 4:5]   # [batch, 3, 26, 26, 1]
        pred_cls = predictions[..., 5:]    # [batch, 3, 26, 26, num_classes]
        
        # 计算损失
        obj_loss = 0
        box_loss = 0
        cls_loss = 0
        
        for batch_idx in range(batch_size):
            batch_targets = targets[batch_idx]
            
            if len(batch_targets) == 0:
                # 如果没有目标，只计算objectness损失（应该是0）
                obj_loss += torch.mean(pred_obj[batch_idx] ** 2)
                continue
            
            # 这里简化处理 - 实际应该进行anchor匹配
            # 为了简化，我们只计算一个简单的MSE损失
            if len(batch_targets) > 0:
                # 使用第一个anchor和第一个目标作为示例
                target_box = batch_targets[0, 1:].unsqueeze(0).unsqueeze(0).unsqueeze(0)
                box_loss += self.mse_loss(pred_boxes[batch_idx, 0:1, 0:1, 0:1], target_box)
                
                # objectness损失
                obj_loss += self.mse_loss(pred_obj[batch_idx, 0:1, 0:1, 0:1], torch.ones_like(pred_obj[batch_idx, 0:1, 0:1, 0:1]))
                
                # 类别损失
                target_cls = torch.zeros_like(pred_cls[batch_idx, 0:1, 0:1, 0:1])
                target_cls[..., 0] = 1  # 只有一个类别
                cls_loss += self.bce_loss(pred_cls[batch_idx, 0:1, 0:1, 0:1], target_cls)
        
        total_loss = box_loss + obj_loss + cls_loss
        return total_loss, box_loss, obj_loss, cls_loss
