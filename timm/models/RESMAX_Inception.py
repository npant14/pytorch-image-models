"""
An implementation of Resmax Inception:
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
from .RESMAX import Residual, S1_Res, S1_Res_Big, S1_Res_Small, S2b_Res, S3_Res, C, C_scoring
from .ALEXMAX3 import C_scoring2
from .HMAX import get_ip_scales
from .ALEXMAX import S1

class Inception(nn.Module):
    def __init__(self, in_channels, out_channels, conv_type='double3x3', **kwargs):
        """
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels per branch
            conv_type: Type of convolution for branch 3 ('use5x5', 'use7x7', or 'double3x3')
        """
        super(Inception, self).__init__(**kwargs)
        
        # Each branch will output out_channels channels 
        branch_channels = out_channels
        
        # Branch 1: 1x1 conv
        self.b1_1 = nn.Conv2d(in_channels, branch_channels, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(branch_channels)
        
        # Branch 2: 1x1 conv -> 3x3 conv
        self.b2_1 = nn.Conv2d(in_channels, branch_channels, kernel_size=1)
        self.bn2_1 = nn.BatchNorm2d(branch_channels)
        self.b2_2 = nn.Conv2d(branch_channels, branch_channels, kernel_size=3, padding=1)
        self.bn2_2 = nn.BatchNorm2d(branch_channels)
        
        # Branch 3: varies based on conv_type
        self.b3_1 = nn.Conv2d(in_channels, branch_channels, kernel_size=1)
        self.bn3_1 = nn.BatchNorm2d(branch_channels)
        
        if conv_type == 'use5x5':
            self.b3_2 = nn.Conv2d(branch_channels, branch_channels, kernel_size=5, padding=2)
            self.bn3_2 = nn.BatchNorm2d(branch_channels)
            self.b3_3 = None
            self.bn3_3 = None
        elif conv_type == 'use7x7':
            self.b3_2 = nn.Conv2d(branch_channels, branch_channels, kernel_size=7, padding=3)
            self.bn3_2 = nn.BatchNorm2d(branch_channels)
            self.b3_3 = None
            self.bn3_3 = None
        elif conv_type == 'double3x3':
            self.b3_2 = nn.Conv2d(branch_channels, branch_channels, kernel_size=3, padding=1)
            self.bn3_2 = nn.BatchNorm2d(branch_channels)
            self.b3_3 = nn.Conv2d(branch_channels, branch_channels, kernel_size=3, padding=1)
            self.bn3_3 = nn.BatchNorm2d(branch_channels)
        else:
            raise ValueError(f"Unsupported conv_type: {conv_type}")
        
        # Branch 4: maxpool -> 1x1 conv
        self.b4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.b4_2 = nn.Conv2d(in_channels, branch_channels, kernel_size=1)
        self.bn4 = nn.BatchNorm2d(branch_channels)

    def forward(self, x):
        # Branch 1
        b1 = F.relu(self.bn1(self.b1_1(x)))
        
        # Branch 2
        b2 = F.relu(self.bn2_1(self.b2_1(x)))
        b2 = F.relu(self.bn2_2(self.b2_2(b2)))
        
        # Branch 3
        b3 = F.relu(self.bn3_1(self.b3_1(x)))
        b3 = F.relu(self.bn3_2(self.b3_2(b3)))
        if self.b3_3 is not None:  # For double3x3
            b3 = F.relu(self.bn3_3(self.b3_3(b3)))
        
        # Branch 4
        b4 = self.b4_1(x)
        b4 = F.relu(self.bn4(self.b4_2(b4)))
        
        # Concatenate along channel dimension
        out = torch.cat([b1, b2, b3, b4], dim=1)
        return out


class S2_Inception(nn.Module):
    def __init__(self, conv_type='double3x3', **kwargs):
        super(S2_Inception, self).__init__()
        self.layer = nn.Sequential(
            Inception(96, 128, conv_type),  # Input from C1 (96 channels)
            Inception(512, 256, conv_type)  # Input channels = 128*4 from previous inception
        )

    def forward(self, x_pyramid):
        # Store the input for bypass connection
        self.bypass = x_pyramid
        out = [self.layer(x) for x in x_pyramid]
        return out


class IncepMAX(nn.Module):
    def __init__(self, num_classes=1000, big_size=322, small_size=227, in_chans=3, 
                 ip_scale_bands=1, classifier_input_size=13312, contrastive_loss=False, pyramid=False, **kwargs):
        super(IncepMAX, self).__init__()
        self.num_classes = num_classes
        self.in_chans = in_chans
        self.contrastive_loss = contrastive_loss
        self.ip_scale_bands = ip_scale_bands
        self.pyramid = pyramid
        self.big_size = big_size
        self.small_size = small_size

        # Keep existing S1 layers
        self.s1_big = S1_Res_Big()
        self.s1_small = S1_Res_Small()
        
        self.c1 = C_scoring(96,
            nn.MaxPool2d(kernel_size=6, stride=3, padding=3),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        
        # Replace S2 with Inception version
        self.s2 = S2_Inception(kwargs['conv_type'])
        self.c2 = C(
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            global_scale_pool=False
        )
        
        # Modified S3 to handle concatenated input
        self.s3 = S3_Res(input_channels=1024)
        self.global_pool = C(global_scale_pool=True)
        
        # Classifier remains the same
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

    def forward(self, x, pyramid=False):
        out = self.make_ip(x)
        
        out_1 = self.s1_big([out[0]])
        out_2 = self.s1_small([out[1]])
        out_c1 = self.c1([out_1[0], out_2[0]])
        
        out = self.s2(out_c1)
        out_c2 = self.c2(out)

        if pyramid:
            return out_c1[0], out_c2[0]

        # pdb.set_trace()
        out = self.s3(out_c2)
        out = self.global_pool(out)
        out = out.reshape(out.size(0), -1)
        
        out = self.fc(out)
        out = self.fc1(out)
        out = self.fc2(out)

        if self.contrastive_loss:
            return out, out_c1[0], out_c2[0]

        return out
    

class IncepMAX_Bypass(nn.Module):
    def __init__(self, num_classes=1000, big_size=322, small_size=227, in_chans=3, 
                 ip_scale_bands=1, classifier_input_size=13312, contrastive_loss=False, pyramid=False, **kwargs):
        super(IncepMAX, self).__init__()
        self.num_classes = num_classes
        self.in_chans = in_chans
        self.contrastive_loss = contrastive_loss
        self.ip_scale_bands = ip_scale_bands
        self.pyramid = pyramid
        self.big_size = big_size
        self.small_size = small_size

        # Keep existing S1 layers
        self.s1_big = S1_Res_Big()
        self.s1_small = S1_Res_Small()
        
        self.c1 = C_scoring(96,
            nn.MaxPool2d(kernel_size=6, stride=3, padding=3),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        
        # Replace S2 with Inception version
        self.s2 = S2_Inception(kwargs['conv_type'])
        self.s2b = S2b_Res()
        self.c2 = C(
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            global_scale_pool=False
        )
        self.c2b_seq = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(1024, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        # Modified S3 to handle concatenated input
        self.s3 = S3_Res(input_channels=1024)
        self.global_pool = C(global_scale_pool=True)
        
        # Classifier remains the same
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

    def forward(self, x, pyramid=False):
        out = self.make_ip(x)

        out_1 = self.s1(out)
        out_c1 = self.c1(out_1)
        
        out = self.s2(out_c1)
        out_c2 = self.c2(out)

        bypass = self.s2b(out_c1)
        bypass = self.c2b_seq(bypass[0])
        bypass = bypass.reshape(bypass.size(0), -1)

        # pdb.set_trace()
        out = self.s3(out_c2)
        out = self.global_pool(out)
        out = out.reshape(out.size(0), -1)

        # merge bypass and main path
        out = torch.cat([out, bypass], dim=1)
        
        out = self.fc(out)
        out = self.fc1(out)
        out = self.fc2(out)

        if self.contrastive_loss:
            return out, out_c1[0], out_c2[0]

        return out


class IncepMAX_C_Score2(nn.Module):
    def __init__(self, num_classes=1000, big_size=322, small_size=227, in_chans=3, 
                 ip_scale_bands=1, classifier_input_size=13312, contrastive_loss=False, pyramid=False, **kwargs):
        super(IncepMAX_C_Score2, self).__init__()
        self.num_classes = num_classes
        self.in_chans = in_chans
        self.contrastive_loss = contrastive_loss
        self.ip_scale_bands = ip_scale_bands
        self.pyramid = pyramid
        self.big_size = big_size
        self.small_size = small_size

        # Keep existing S1 layers
        self.s1 = S1_Res()
        
        self.c1= C_scoring2(
            96,
            nn.MaxPool2d(kernel_size = 3, stride = 2), 
            nn.MaxPool2d(kernel_size = 4, stride = 3), 
            global_scale_pool=False
        )
        
        # Replace S2 with Inception version
        self.s2 = S2_Inception(kwargs['conv_type'])
        self.c2 = C(
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            global_scale_pool=False
        )
        
        # Modified S3 to handle concatenated input
        self.s3 = S3_Res(input_channels=1024)
        self.global_pool = C(global_scale_pool=True)
        
        # Classifier remains the same
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
        ## num_scale_bands = num images in IP - 1
        num_scale_bands = self.ip_scale_bands # 1
        base_image_size = int(x.shape[-1])
        scale = 4   ## factor in exponenet

        image_scales = get_ip_scales(num_scale_bands, base_image_size, scale)
        
        if len(image_scales) > 1:
            image_pyramid = []
            for i_s in image_scales:
                i_s = int(i_s)
                interpolated_img = F.interpolate(x, size = (i_s, i_s), mode = 'bilinear')

                image_pyramid.append(interpolated_img)
            return image_pyramid
        else: 
            return [x]

    def forward(self, x, pyramid=False):
        out = self.make_ip(x)
        
        out_1 = self.s1(out)
        out_c1 = self.c1(out_1)
        
        out = self.s2(out_c1)
        out_c2 = self.c2(out)

        if pyramid:
            return out_c1[0], out_c2[0]

        # pdb.set_trace()
        out = self.s3(out_c2)
        out = self.global_pool(out)
        out = out.reshape(out.size(0), -1)
        
        out = self.fc(out)
        out = self.fc1(out)
        out = self.fc2(out)

        if self.contrastive_loss:
            return out, out_c1[0], out_c2[0]

        return out
    

class IncepMAX_C_Score2(nn.Module):
    def __init__(self, num_classes=1000, big_size=322, small_size=227, in_chans=3, 
                 ip_scale_bands=1, classifier_input_size=13312, contrastive_loss=False, pyramid=False, **kwargs):
        super(IncepMAX_C_Score2, self).__init__()
        self.num_classes = num_classes
        self.in_chans = in_chans
        self.contrastive_loss = contrastive_loss
        self.ip_scale_bands = ip_scale_bands
        self.pyramid = pyramid
        self.big_size = big_size
        self.small_size = small_size

        # Keep existing S1 layers
        self.s1 = S1_Res()
        if kwargs['s1'] == 'conv_11x11':
            self.s1 = S1(kernel_size=11, stride=4, padding=0)
        
        self.c1= C_scoring2(
            96,
            nn.MaxPool2d(kernel_size = 3, stride = 2), 
            nn.MaxPool2d(kernel_size = 4, stride = 3), 
            global_scale_pool=False
        )
        
        # Replace S2 with Inception version
        self.s2 = S2_Inception(kwargs['conv_type'])
        self.c2 = C(
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            global_scale_pool=False
        )
        
        # Modified S3 to handle concatenated input
        self.s3 = S3_Res(input_channels=1024)
        self.global_pool = C(global_scale_pool=True)
        
        # Classifier remains the same
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
        ## num_scale_bands = num images in IP - 1
        num_scale_bands = self.ip_scale_bands # 1
        base_image_size = int(x.shape[-1])
        scale = 4   ## factor in exponenet

        image_scales = get_ip_scales(num_scale_bands, base_image_size, scale)
        
        if len(image_scales) > 1:
            image_pyramid = []
            for i_s in image_scales:
                i_s = int(i_s)
                interpolated_img = F.interpolate(x, size = (i_s, i_s), mode = 'bilinear')

                image_pyramid.append(interpolated_img)
            return image_pyramid
        else: 
            return [x]

    def forward(self, x, pyramid=False):
        out = self.make_ip(x)
        
        out_1 = self.s1(out)
        out_c1 = self.c1(out_1)
        
        out = self.s2(out_c1)
        out_c2 = self.c2(out)

        if pyramid:
            return out_c1[0], out_c2[0]

        # pdb.set_trace()
        out = self.s3(out_c2)
        out = self.global_pool(out)
        out = out.reshape(out.size(0), -1)
        
        out = self.fc(out)
        out = self.fc1(out)
        out = self.fc2(out)

        if self.contrastive_loss:
            return out, out_c1[0], out_c2[0]

        return out


class IncepMAX_Bypass_C_Score2(nn.Module):
    def __init__(self, num_classes=1000, big_size=322, small_size=227, in_chans=3, 
                 ip_scale_bands=1, classifier_input_size=13312, contrastive_loss=False, pyramid=False, **kwargs):
        super(IncepMAX_Bypass_C_Score2, self).__init__()
        self.num_classes = num_classes
        self.in_chans = in_chans
        self.contrastive_loss = contrastive_loss
        self.ip_scale_bands = ip_scale_bands
        self.pyramid = pyramid
        self.big_size = big_size
        self.small_size = small_size

        # Keep existing S1 layers
        self.s1 = S1_Res()
        
        self.c1= C_scoring2(
            96,
            nn.MaxPool2d(kernel_size = 3, stride = 2), 
            nn.MaxPool2d(kernel_size = 4, stride = 3), 
            global_scale_pool=False
        )
        
        # Replace S2 with Inception version
        self.s2 = S2_Inception(kwargs['conv_type'])
        self.s2b = S2b_Res()
        self.c2 = C(
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            global_scale_pool=False
        )
        self.c2b_seq = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(1024, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        # Modified S3 to handle concatenated input
        self.s3 = S3_Res(input_channels=1024)
        self.global_pool = C(global_scale_pool=True)
        
        # Classifier remains the same
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
        ## num_scale_bands = num images in IP - 1
        num_scale_bands = self.ip_scale_bands # 1
        base_image_size = int(x.shape[-1])
        scale = 4   ## factor in exponenet

        image_scales = get_ip_scales(num_scale_bands, base_image_size, scale)
        
        if len(image_scales) > 1:
            image_pyramid = []
            for i_s in image_scales:
                i_s = int(i_s)
                interpolated_img = F.interpolate(x, size = (i_s, i_s), mode = 'bilinear')

                image_pyramid.append(interpolated_img)
            return image_pyramid
        else: 
            return [x]

    def forward(self, x, pyramid=False):
        out = self.make_ip(x)
        
        out_1 = self.s1(out)
        out_c1 = self.c1(out_1)
        
        out = self.s2(out_c1)
        out_c2 = self.c2(out)

        bypass = self.s2b(out_c1)
        bypass = self.c2b_seq(bypass[0])
        bypass = bypass.reshape(bypass.size(0), -1)

        # pdb.set_trace()
        out = self.s3(out_c2)
        out = self.global_pool(out)
        out = out.reshape(out.size(0), -1)

        # merge bypass and main path
        out = torch.cat([out, bypass], dim=1)
        # out += bypass
        
        out = self.fc(out)
        out = self.fc1(out)
        out = self.fc2(out)

        if self.contrastive_loss:
            return out, out_c1[0], out_c2[0]

        return out
    
@register_model
def incepmax(pretrained=False, **kwargs):
    #deleting some kwargs that are messing up training
    try:
        del kwargs["pretrained_cfg"]
        del kwargs["pretrained_cfg_overlay"]
        del kwargs["drop_rate"]
    except:
        pass
    model = IncepMAX(**kwargs)
    if pretrained:
        raise NotImplementedError
    return model


@register_model
def incepmax_c_score2(pretrained=False, **kwargs):
    #deleting some kwargs that are messing up training
    try:
        del kwargs["pretrained_cfg"]
        del kwargs["pretrained_cfg_overlay"]
        del kwargs["drop_rate"]
    except:
        pass
    model = IncepMAX_C_Score2(**kwargs)
    if pretrained:
        raise NotImplementedError
    return model

@register_model
def incepmax_bypass_c_score2(pretrained=False, **kwargs):
    #deleting some kwargs that are messing up training
    try:
        del kwargs["pretrained_cfg"]
        del kwargs["pretrained_cfg_overlay"]
        del kwargs["drop_rate"]
    except:
        pass
    model = IncepMAX_Bypass_C_Score2(**kwargs)
    if pretrained:
        raise NotImplementedError
    return model