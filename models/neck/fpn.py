import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from torch.nn.modules.batchnorm import _BatchNorm
from models.bn_lib.nn.modules import SynchronizedBatchNorm2d
norm_layer = partial(SynchronizedBatchNorm2d, momentum=0.9)
import math


class External_attention(nn.Module):
    '''
    Arguments:
        c (int): The input and output channel number.
    '''
    def __init__(self, c):
        super(External_attention, self).__init__()
        
        self.conv1 = nn.Conv2d(c, c, 1)

        self.k = 64
        self.linear_0 = nn.Conv1d(c, self.k, 1, bias=False)

        self.linear_1 = nn.Conv1d(self.k, c, 1, bias=False)
        self.linear_1.weight.data = self.linear_0.weight.data.permute(1, 0, 2)        
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(c, c, 1, bias=False),
            norm_layer(c))        
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, _BatchNorm):
                m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()
 

    def forward(self, x):
        idn = x
        x = self.conv1(x)

        b, c, h, w = x.size()
        n = h*w
        x = x.view(b, c, h*w)   # b * c * n 

        attn = self.linear_0(x) # b, k, n
        attn = F.softmax(attn, dim=-1) # b, k, n

        attn = attn / (1e-9 + attn.sum(dim=1, keepdim=True)) #  # b, k, n
        x = self.linear_1(attn) # b, c, n

        x = x.view(b, c, h, w)
        x = self.conv2(x)
        x = x + idn
        x = F.relu(x)
        return x

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio = 4, flag=True):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.conv1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.flag = flag
        self.sigmoid = nn.Sigmoid()

        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)

    def forward(self, x):
        avg_out = self.conv2(self.relu(self.conv1(self.avg_pool(x))))
        max_out = self.conv2(self.relu(self.conv1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)*x if self.flag else self.sigmoid(out)
    
class DimensionMatchingModule(nn.Module):
    """维度匹配模块"""
    def __init__(self, in_channels):
        super(DimensionMatchingModule, self).__init__()
        self.conv1x1 = nn.Conv2d(in_channels, 256, kernel_size=1)
        #nn.init.kaiming_uniform_(self.conv1x1.weight, nonlinearity='relu')
        if self.conv1x1.bias is not None:
            nn.init.constant_(self.conv1x1.bias, 0)
    def forward(self, x):
        return self.conv1x1(x)

class SFFModule(nn.Module):
    """特征融合模块"""
    def __init__(self, in_channels):
        super(SFFModule, self).__init__()
        self.ca_module = ChannelAttention(in_channels)
        self.deconv = nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=1, output_padding=1)

        #self.matching_module = DimensionMatchingModule(in_channels)

    def forward(self, f_high, f_low):
        # 动态获取f_low的尺寸作为目标尺寸
        target_size = f_low.shape[2:]
        # 调整高层特征图的大小以匹配目标尺寸
        f_high_ = self.deconv(f_high)
        f_high_resized = F.interpolate(f_high_, size=target_size, mode='bilinear', align_corners=True)
        # 计算注意力权重
        ca_weight = self.ca_module(f_high_resized)
        # 使用维度匹配模块统一特征图的通道数
        # f_high_matched = self.matching_module(f_high_resized)
        # f_low_matched = self.matching_module(f_low)
        # 应用注意力权重并融合特征图
        #f_out = f_low_matched * ca_weight + f_high_matched
        f_out = f_low * ca_weight + f_high_resized
        return f_out

class FPN(nn.Module):
    """特征金字塔网络，调整以使用后三层特征图"""
    def __init__(self, in_channel_list=[128, 256, 512]):
        super(FPN, self).__init__()
        # 因为只在最后一层使用特征选择模块，所以我们只为最后一层初始化它
        #self.last_layer_fsm = ChannelAttention(in_channel_list[-1])
        self.last_layer_fsm = External_attention(in_channel_list[-1])
        self.dmm4 = DimensionMatchingModule(512)
        self.dmm3 = DimensionMatchingModule(128)

        self.sff = SFFModule(256)    

    def forward(self, features):
        # 假设features是从主干网络后三层得到的特征图列表：[S3, S4, S5]
        # (128,50,50)  (256,100,100) (512,200,200)
        S3, S4, S5 = features
        S5 = self.last_layer_fsm(S5)
        S5 = self.dmm4(S5)
        S3 = self.dmm3(S3)

        S4_ = self.sff(S5, S4)
        S3_ = self.sff(S4_, S3)
        return S3_