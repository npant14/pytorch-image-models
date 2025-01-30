"""
An implementation of RESMAX:
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy as sp
import time
import pdb

from ._builder import build_model_with_cfg
from ._manipulate import checkpoint_seq
from ._registry import register_model, generate_default_cfgs
from .ALEXMAX import C_scoring, C

class Residual(nn.Module):
    def __init__(self, in_channels, out_channels, strides=1):
        
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels,
                               kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=3, padding=1)
        
        if strides > 1 or in_channels != out_channels:
            self.conv3 = nn.Conv2d(in_channels, out_channels,
                                   kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)
    

class S1_Res_Big(nn.Module):
    def __init__(self):
        super(S1_Res_Big, self).__init__()
        self.layer1 = nn.Sequential(
            Residual(3, 48, strides=2),
            Residual(48, 48),
            Residual(48, 96, strides=2)
        )

    def forward(self, x_pyramid):
        return [self.layer1(x) for x in x_pyramid]

class S1_Res_Small(nn.Module):
    def __init__(self):
        super(S1_Res_Small, self).__init__()
        self.layer1 = nn.Sequential(
            Residual(3, 48, strides=2),
            Residual(48, 96, strides=2)
        )

    def forward(self, x_pyramid):
        return [self.layer1(x) for x in x_pyramid]

class S2_Res(nn.Module):
    def __init__(self):
        super(S2_Res, self).__init__()
        self.layer = nn.Sequential(
            Residual(96, 128),
            Residual(128, 256)
        )

    def forward(self, x_pyramid):
        return [self.layer(x) for x in x_pyramid]

class S3_Res(nn.Module):
    def __init__(self, input_channels=256):
        super(S3_Res, self).__init__()
        self.layer = nn.Sequential(
            Residual(input_channels, 256),
            Residual(256, 384),
            Residual(384, 384),
            Residual(384, 256)
        )

    def forward(self, x_pyramid):
        return [self.layer(x) for x in x_pyramid]
    

class RESMAX(nn.Module):
    def __init__(self, num_classes=1000, big_size=322, small_size=227, in_chans=3, 
                 ip_scale_bands=1, classifier_input_size=13312, contrastive_loss=False, pyramid=False, **kwargs):
        self.num_classes = num_classes
        self.in_chans = in_chans
        self.contrastive_loss = contrastive_loss
        self.ip_scale_bands = ip_scale_bands
        self.pyramid = pyramid
        self.big_size = big_size
        self.small_size = small_size
        super(RESMAX, self).__init__()

        self.s1_big = S1_Res_Big()
        self.s1_small = S1_Res_Small()
        
        self.c1 = C_scoring(96,
            nn.MaxPool2d(kernel_size=6, stride=3, padding=3),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        
        self.s2 = S2_Res()
        self.c2 = C(
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            global_scale_pool=False
        )
        
        self.s3 = S3_Res()
        self.global_pool = C(global_scale_pool=True)
        
        # Keep classifier layers the same
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(classifier_input_size, 4096),
            nn.ReLU()
        )
        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(4096, num_classes)
        )

    def make_ip(self, x):
        image_scales = [self.big_size, self.small_size]

        if len(image_scales) > 1:
            image_pyramid = []
            for i_s in image_scales:
                i_s = int(i_s)
                interpolated_img = F.interpolate(x, size=(i_s, i_s), mode='bilinear')
                image_pyramid.append(interpolated_img)
            return image_pyramid
        else:
            return [x]

    def forward(self, x,pyramid=False):
        #resize image
        # always making pyramid to start with only two scales. 
        out = self.make_ip(x)
        
        ## should make SxBxCxHxW
        out_1 = self.s1_big([out[0]])
        out_2 = self.s1_small([out[1]])
        out_c1 = self.c1([out_1[0],out_2[0]])
        #bypass layers
        out = self.s2(out_c1)
        out_c2 = self.c2(out)

        if pyramid:
            return out_c1[0],out_c2[0]

        out = self.s3(out_c2)
        out = self.global_pool(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.fc1(out)
        out = self.fc2(out)

        if self.contrastive_loss:
            return out, out_c1[0],out_c2[0]

        return out
    

@register_model
def resmax(pretrained=False, **kwargs):
    #deleting some kwargs that are messing up training
    try:
        del kwargs["pretrained_cfg"]
        del kwargs["pretrained_cfg_overlay"]
        del kwargs["drop_rate"]
    except:
        pass
    model = RESMAX(**kwargs)
    if pretrained:
        raise NotImplementedError
    return model




