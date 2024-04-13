import torch
import torch.nn as nn
import torch.nn.functional as F

class BalanceCrossEntropyLoss(nn.Module):
    def __init__(self, loss_weight=1.0, pos_weight=4.0):
        super(BalanceCrossEntropyLoss, self).__init__()
        self.loss_weight = loss_weight
        # 初始化 pos_weight 为一个张量
        self.pos_weight = torch.tensor([pos_weight])
        
        # 初始化 BCEWithLogitsLoss，内置了 sigmoid 激活
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight, reduction='none')

    def forward(self, input, target, mask, reduce=True):
        batch_size = input.size(0)

        input = input.view(batch_size, -1)
        target = target.view(batch_size, -1).float()
        mask = mask.view(batch_size, -1).float()

        # 计算 BCEWithLogitsLoss 损失
        loss = self.loss_fn(input, target)
        
        # 应用 mask
        loss = loss * mask
        
        # 应用 loss_weight 和 reduce 选项
        if reduce:
            # 只考虑非零元素计算平均损失
            loss = torch.mean(loss[loss != 0])
        
        return self.loss_weight * loss