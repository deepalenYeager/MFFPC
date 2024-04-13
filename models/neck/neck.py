import torch.nn as nn
from models.utils.nas_utils import set_layer_from_config
import torch.nn.functional as F
import torch
import json
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


class HSFPN(nn.Module):
    def __init__(self, in_planes, ratio = 4, flag=True):
        super(HSFPN, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.conv1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.conv3 = nn.Conv2d(in_planes, in_planes, 1, bias=False)  # 1x1 convolution layer added
        self.flag = flag
        self.sigmoid = nn.Sigmoid()
        

        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)

    def forward(self, x):
        avg_out = self.conv2(self.relu(self.conv1(self.avg_pool(x))))
        max_out = self.conv2(self.relu(self.conv1(self.max_pool(x))))
        out = avg_out + max_out
        attention_out = self.sigmoid(out) * x  # Apply sigmoid attention
        if self.flag:
            return self.conv3(attention_out)  # Pass through 1x1 convolution before returning
        else:
            return self.sigmoid(out)  # Return attention scores directly if flag is False


class Neck(nn.Module):
    def __init__(self, reduce_layer1, reduce_layer2, reduce_layer3, reduce_layer4, in_planes=[64, 128, 256, 512]):
        super(Neck, self).__init__()
        self.reduce_layer1 = reduce_layer1
        self.reduce_layer2 = reduce_layer2
        self.reduce_layer3 = reduce_layer3
        self.reduce_layer4 = reduce_layer4
        self.ex = External_attention(128)

        #Instantiate HSFPN modules for each reduced feature map
        # self.hsfpn1 = HSFPN()
        # self.hsfpn2 = HSFPN()
        # 

        
        self.hsfpn3 = HSFPN(in_planes=in_planes[2])
        self.hsfpn4 = HSFPN(in_planes=in_planes[3])

        self._initialize_weights()
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    
    def _upsample(self, x, y):
        _, _, H, W = y.size()
        return F.upsample(x, size=(H, W), mode='bilinear')
    
    def forward(self, x):
        f1, f2, f3, f4 = x
        """
        f1: torch.Size([16, 64, 160, 160]) 
        f2: torch.Size([16, 128, 80, 80]) 
        f3: torch.Size([16, 256, 40, 40]) 
        f4: torch.Size([16, 512, 20, 20])
        """
        f1 = self.reduce_layer1(f1)
        f2 = self.reduce_layer2(f2)
        f3 = self.reduce_layer3(self.hsfpn3(f3))
        f4 = self.reduce_layer4(self.hsfpn4(f4))
        
        # f1 = self.reduce_layer1(f1)
        # f2 = self.reduce_layer2(f2)
        # f3 = self.reduce_layer3(f3)
        # f4 = self.reduce_layer4(f4)
        f4 = self.ex(f4)
        f2 = self._upsample(f2, f1)
        f3 = self._upsample(f3, f1)
        f4 = self._upsample(f4, f1)
        """
        f1: torch.Size([16, 128, 160, 160]) 
        f2: torch.Size([16, 128, 160, 160]) 
        f3: torch.Size([16, 128, 160, 160]) 
        f4: torch.Size([16, 128, 160, 160])
        """
        f = torch.cat((f1, f2, f3, f4), 1)
        """
        f: torch.Size([16, 512, 160, 160])  384
        """
        return f

    @staticmethod
    def build_from_config(config):
        reduce_layer1 = set_layer_from_config(config['reduce_layer1'])
        reduce_layer2 = set_layer_from_config(config['reduce_layer2'])
        reduce_layer3 = set_layer_from_config(config['reduce_layer3'])
        reduce_layer4 = set_layer_from_config(config['reduce_layer4'])
        return MFFPCNeck(reduce_layer1, reduce_layer2, reduce_layer3, reduce_layer4)
    
    
def MFFPC_neck(config, **kwargs):
    neck_config = json.load(open(config, 'r'))['neck']
    neck = MFFPCNeck.build_from_config(neck_config, **kwargs)
    return neck


# import torch.nn as nn
# from models.utils.nas_utils import set_layer_from_config
# import torch.nn.functional as F
# import torch
# import json


# class MFFPCNeck(nn.Module):
#     def __init__(self, reduce_layer1, reduce_layer2, reduce_layer3, reduce_layer4):
#         super(MFFPCNeck, self).__init__()
#         self.reduce_layer1 = reduce_layer1
#         self.reduce_layer2 = reduce_layer2
#         self.reduce_layer3 = reduce_layer3
#         self.reduce_layer4 = reduce_layer4

#         self._initialize_weights()

#     def _initialize_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight)
#             elif isinstance(m, nn.BatchNorm2d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()
    
#     def _upsample(self, x, y):
#         _, _, H, W = y.size()
#         return F.upsample(x, size=(H, W), mode='bilinear')
    
#     def forward(self, x):
#         f1, f2, f3, f4 = x
#         f1 = self.reduce_layer1(f1)
#         f2 = self.reduce_layer2(f2)
#         f3 = self.reduce_layer3(f3)
#         f4 = self.reduce_layer4(f4)

#         f2 = self._upsample(f2, f1)
#         f3 = self._upsample(f3, f1)
#         f4 = self._upsample(f4, f1)
#         f = torch.cat((f1, f2, f3, f4), 1)
#         return f

#     @staticmethod
#     def build_from_config(config):
#         reduce_layer1 = set_layer_from_config(config['reduce_layer1'])
#         reduce_layer2 = set_layer_from_config(config['reduce_layer2'])
#         reduce_layer3 = set_layer_from_config(config['reduce_layer3'])
#         reduce_layer4 = set_layer_from_config(config['reduce_layer4'])
#         return MFFPCNeck(reduce_layer1, reduce_layer2, reduce_layer3, reduce_layer4)
    
    
# def MFFPC_neck(config, **kwargs):
#     neck_config = json.load(open(config, 'r'))['neck']
#     neck = MFFPCNeck.build_from_config(neck_config, **kwargs)
#     return neck