"""
An implementation of RESMAX:
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
import scipy as sp
import time
import pdb
import torchvision
import numpy as np
import random
import logging
import os
from torchvision.utils import save_image


from ._builder import build_model_with_cfg
from ._manipulate import checkpoint_seq
from ._registry import register_model, generate_default_cfgs
from .ALEXMAX import C_scoring, C
from .ALEXMAX3 import C_scoring2
from .ALEXMAX3_optimized import C_scoring2_optimized, C_scoring2_optimized_debug
from .HMAX import get_ip_scales

def pad_to_size(a, size, mode='constant'):
    current_size = (a.shape[-2], a.shape[-1])
    total_pad_h = size[0] - current_size[0]
    pad_top = total_pad_h // 2
    pad_bottom = total_pad_h - pad_top

    total_pad_w = size[1] - current_size[1]
    pad_left = total_pad_w // 2
    pad_right = total_pad_w - pad_left

    a = nn.functional.pad(a, (pad_left, pad_right, pad_top, pad_bottom), mode=mode)

    return a


class Residual(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, strides=1):
        
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels,
                               kernel_size=kernel_size, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=kernel_size, padding=1)
        
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
    
class S1_Res(nn.Module):
    def __init__(self):
        super(S1_Res, self).__init__()
        self.layer1 = nn.Sequential(
            Residual(3, 48, strides=2),
            Residual(48, 48),
            Residual(48, 96, strides=2)
        )

    def forward(self, x_pyramid):
        if type(x_pyramid) == list:
            return [self.layer1(x) for x in x_pyramid]
        else:
            return self.layer1(x_pyramid)


class S2_Res(nn.Module):
    def __init__(self):
        super(S2_Res, self).__init__()
        self.layer = nn.Sequential(
            Residual(96, 128),
            Residual(128, 256)
        )

    def forward(self, x_pyramid):
        return [self.layer(x) for x in x_pyramid]
    

class S2b_Res(nn.Module):
    def __init__(self):
        super(S2b_Res, self).__init__()
        # Each Residual block has 2 3x3 convs, so we need fewer blocks
        # to achieve same receptive field
        # 4x4 -> 1 residual block (2 3x3 convs)
        # 8x8 -> 2 residual blocks (4 3x3 convs)
        # 12x12 -> 3 residual blocks (6 3x3 convs)
        # 16x16 -> 4 residual blocks (8 3x3 convs)
        self.kernel_to_blocks = {4: 1, 8: 2, 12: 3, 16: 4}
        
        self.s2b_seqs = nn.ModuleList()
        for kernel_size, num_blocks in self.kernel_to_blocks.items():
            blocks = []
            # Initial projection to higher dimensions
            blocks.append(nn.Sequential(
                nn.Conv2d(96, 256, kernel_size=1),  # 1x1 conv for dimension matching
                nn.BatchNorm2d(256),
                nn.ReLU(True)
            ))
            # Stack of residual blocks
            for _ in range(num_blocks):
                blocks.append(Residual(256, 256))
                
            self.s2b_seqs.append(nn.Sequential(*blocks))

    def forward(self, x_pyramid):
        bypass = [torch.cat([seq(out) for seq in self.s2b_seqs], dim=1) for out in x_pyramid]
        return bypass


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


class RESMAX_V2(nn.Module):
    def __init__(self, num_classes=1000, big_size=322, small_size=227, in_chans=3, 
                 ip_scale_bands=1, classifier_input_size=13312, contrastive_loss=False, pyramid=False,
                 bypass=False, main_route=False,
                 c_scoring='v2',
                 **kwargs):
        self.num_classes = num_classes
        self.in_chans = in_chans
        self.contrastive_loss = contrastive_loss
        self.ip_scale_bands = ip_scale_bands
        self.pyramid = pyramid
        self.big_size = big_size
        self.small_size = small_size
        self.bypass = bypass
        self.c_scoring = c_scoring
        self.main_route = main_route
        super(RESMAX_V2, self).__init__()

        self.s1 = S1_Res()

        # C1 using optimized layer
        self.c1 = C_scoring2_optimized(
            num_channels=96,
            pool_func1=nn.MaxPool2d(kernel_size=3, stride=2),
            pool_func2=nn.MaxPool2d(kernel_size=4, stride=3),
            skip=1,
            global_scale_pool=False
        )
        
        self.s2 = S2_Res()
        # C2 using optimized layer
        self.c2 = C_scoring2_optimized(
            num_channels=256,
            pool_func1=nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            pool_func2=nn.MaxPool2d(kernel_size=6, stride=2),
            resize_kernel_1=3,
            resize_kernel_2=1,
            skip=2,
            global_scale_pool=False
        )
        
        if self.bypass:
            self.s2b = S2b_Res()
            self.c2b_seq = nn.Sequential(
                nn.MaxPool2d(kernel_size=3, stride=2),
                nn.MaxPool2d(kernel_size=3, stride=2),
                nn.Conv2d(1024, 256, kernel_size=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True)
            )

        self.s3 = S3_Res()
        if self.ip_scale_bands > 4:
            self.global_pool = C_scoring2_optimized(
                num_channels=256,
                pool_func1=nn.MaxPool2d(kernel_size=3, stride=2),
                pool_func2=nn.MaxPool2d(kernel_size=6, stride=3, padding=1),
                resize_kernel_1=3,
                resize_kernel_2=1,
                skip=2,
                global_scale_pool=False
            )
        else:
            self.global_pool = C(global_scale_pool=True)

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

        self.print_param_stats()

    def make_ip(self, x, num_scale_bands):
        """
        Build an image pyramid.
        num_scale_bands = number of images in the pyramid - 1
        """
        base_image_size = int(x.shape[-1])
        scale_factor = 4  # exponent factor for scaling
        image_scales = get_ip_scales(num_scale_bands, base_image_size, scale_factor)
        
        if len(image_scales) > 1:
            image_pyramid = []
            for i_s in image_scales:
                i_s = int(i_s)
                interp_img = F.interpolate(x, size=(i_s, i_s), mode='bilinear', align_corners=False)
                image_pyramid.append(interp_img)
            return image_pyramid
        else:
            return [x]

    def forward(self, x, pyramid=False, main_route=False):
        if main_route:
            out = self.make_ip(x, 2)
        else:
            out = self.make_ip(x, self.ip_scale_bands)
        
        out = self.s1(out)
        out_c1 = self.c1(out)
        out = self.s2(out_c1)
        out_c2 = self.c2(out)
        
        if self.bypass:
            bypass = self.s2b(out_c1)
            bypass = self.c2b_seq(bypass[0])
            bypass = bypass.reshape(bypass.size(0), -1)
        
        out = self.s3(out_c2)
        out = self.global_pool(out)
        if isinstance(out, list):
            out = out[0]
        out = out.reshape(out.size(0), -1)

        if self.bypass:
            out = torch.cat([out, bypass], dim=1)
        
        out = self.fc(out)
        out = self.fc1(out)
        out = self.fc2(out)

        if self.contrastive_loss:
            if self.bypass:
                return out, out_c1, out_c2, bypass
            else:
                return out, out_c1, out_c2

        return out
    
    def print_param_stats(self):
        print(f"\nParameter breakdown for {self.__class__.__name__}:\n")
        total_params = 0
        stats = []
        for name, module in self.named_children():
            n_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
            total_params += n_params
            stats.append((name, n_params))

        stats.sort(key=lambda x: x[1], reverse=True)
        print(f"{'Module':30s} | {'# Params':>10s} | {'% of Total':>10s}")
        print("-" * 60)
        for name, count in stats:
            pct = 100 * count / total_params
            print(f"{name:30s} | {count:10,d} | {pct:10.2f}%")
        print("-" * 60)
        print(f"{'Total':30s} | {total_params:10,d} | {100.00:10.2f}%\n")



class S2b_Res1(nn.Module):
    def __init__(self):
        super(S2b_Res1, self).__init__()
        # Each Residual block has 2 3x3 convs, so we need fewer blocks
        # to achieve same receptive field
        # 4x4 -> 1 residual block (2 3x3 convs)
        # 8x8 -> 2 residual blocks (4 3x3 convs)
        # 12x12 -> 3 residual blocks (6 3x3 convs)
        # 16x16 -> 4 residual blocks (8 3x3 convs)
        self.kernel_to_blocks = {4: 1, 8: 2, 12: 3, 16: 4}
        
        self.s2b_seqs = nn.ModuleList()
        for kernel_size, num_blocks in self.kernel_to_blocks.items():
            blocks = []
            # Initial projection to higher dimensions
            blocks.append(nn.Sequential(
                nn.Conv2d(64, 256, kernel_size=1),  # 1x1 conv for dimension matching
                nn.BatchNorm2d(256),
                nn.ReLU(True)
            ))
            # Stack of residual blocks
            for _ in range(num_blocks):
                blocks.append(Residual1(256, 256))

            self.s2b_seqs.append(nn.Sequential(*blocks))

    def forward(self, x_pyramid):
        bypass = [torch.cat([seq(out) for seq in self.s2b_seqs], dim=1) for out in x_pyramid]
        return bypass
    
class Residual1(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, strides=1):
        
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels,
                               kernel_size=kernel_size, padding=1, stride=strides, bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=kernel_size, padding=1, bias=False)
        
        if strides > 1 or in_channels != out_channels:
            self.conv3 = nn.Conv2d(in_channels, out_channels,
                                   kernel_size=1, stride=strides, bias=False)
            self.bn3 = nn.BatchNorm2d(out_channels)
        else:
            self.conv3 = None
            self.bn3 = None

        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3 is not None and self.bn3 is not None:
            X = self.bn3(self.conv3(X))
        Y += X
        return F.relu(Y)

import torch
import torch.nn as nn
import torch.nn.functional as F

class C_adp(nn.Module):
    # Spatial then Scale
    def __init__(self,
                 pool_func1=nn.MaxPool2d(kernel_size=3, stride=2),
                 pool_func2=nn.MaxPool2d(kernel_size=4, stride=3),
                 global_scale_pool=None):
        super(C_adp, self).__init__()
        self.pool1 = pool_func1
        self.pool2 = pool_func2
        self.global_scale_pool = global_scale_pool

    def forward(self, x_pyramid):
        out = []

        if self.global_scale_pool is not None:
            pooled = [self.global_scale_pool(x) for x in x_pyramid]
            out = pooled[0]
            for p in pooled[1:]:
                out = torch.max(out, p)

        else:
            if len(x_pyramid) == 1:
                return [self.pool1(x_pyramid[0])]

            for i in range(len(x_pyramid) - 1):
                x_1 = self.pool1(x_pyramid[i])
                x_2 = self.pool2(x_pyramid[i + 1])

                # Interpolate to match spatial sizes
                if x_1.shape[-1] > x_2.shape[-1]:
                    x_2 = F.interpolate(x_2, size=x_1.shape[-2:], mode='bilinear')
                else:
                    x_1 = F.interpolate(x_1, size=x_2.shape[-2:], mode='bilinear')

                stacked = torch.stack([x_1, x_2], dim=-1)
                to_append, _ = torch.max(stacked, dim=-1)
                out.append(to_append)

        return out

class RESMAX_V3(nn.Module):
    def __init__(self, num_classes=1000, in_chans=3, 
                 ip_scale_bands=1, classifier_input_size=512, contrastive_loss=False, pyramid=False,
                 bypass=False, main_route=False,
                 c_scoring='v2',
                 **kwargs):
        self.num_classes = num_classes
        self.in_chans = in_chans
        self.contrastive_loss = contrastive_loss
        self.ip_scale_bands = ip_scale_bands
        self.pyramid = pyramid
        self.bypass = bypass
        self.c_scoring = c_scoring
        self.main_route = main_route
        self.classifier_input_size = classifier_input_size
        
        super(RESMAX_V3, self).__init__()

        self.s1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        # C1 using optimized layer
        self.c1 = C_scoring2_optimized(
            num_channels=64,
            pool_func1=nn.MaxPool2d(kernel_size=3, stride=2),
            pool_func2=nn.MaxPool2d(kernel_size=4, stride=3),
            skip=1,
            global_scale_pool=False
        )
        
        self.s2 = nn.Sequential(
            Residual1(64, 64),
            Residual1(64, 64)
        )

        # C2 using optimized layer
        self.c2 = C_scoring2_optimized(
            num_channels=64,
            pool_func1=nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            pool_func2=nn.MaxPool2d(kernel_size=6, stride=2),
            resize_kernel_1=3,
            resize_kernel_2=1,
            skip=2,
            global_scale_pool=False
        )
        
        if self.bypass:
            self.s2b = S2b_Res1()
            self.c2b_seq = nn.Sequential(
                nn.MaxPool2d(kernel_size=3, stride=2),
                nn.MaxPool2d(kernel_size=3, stride=2),
                nn.Conv2d(1024, 512, kernel_size=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1))
            )

        self.s3 = nn.Sequential(
            Residual1(64, 128, strides=2),
            Residual1(128, 128),
            Residual1(128, 256, strides=2),
            Residual1(256, 256),
            Residual1(256, 512, strides=2),
            Residual1(512, 512)
        )

        if self.ip_scale_bands > 4:
            self.global_pool = C_scoring2_optimized(
                num_channels=256,
                pool_func1=nn.MaxPool2d(kernel_size=3, stride=2),
                pool_func2=nn.MaxPool2d(kernel_size=6, stride=3, padding=1),
                resize_kernel_1=3,
                resize_kernel_2=1,
                skip=2,
                global_scale_pool=False
            )
        else:
            self.global_pool = C_adp(global_scale_pool=nn.AdaptiveAvgPool2d((1, 1)))

        if self.bypass:
            self.classifier_input_size = classifier_input_size * 2

        self.fc= nn.Sequential(
            nn.Linear(self.classifier_input_size, num_classes)
        )

        self.print_param_stats()

    def make_ip(self, x, num_scale_bands):
        """
        Build an image pyramid.
        num_scale_bands = number of images in the pyramid - 1
        """
        base_image_size = int(x.shape[-1])
        scale_factor = 4  # exponent factor for scaling
        image_scales = get_ip_scales(num_scale_bands, base_image_size, scale_factor)
        
        if len(image_scales) > 1:
            image_pyramid = []
            for i_s in image_scales:
                i_s = int(i_s)
                interp_img = F.interpolate(x, size=(i_s, i_s), mode='bilinear', align_corners=False)
                image_pyramid.append(interp_img)
            return image_pyramid
        else:
            return [x]

    def forward(self, x):
        def apply(module, x):
            return [module(xi) for xi in x] if isinstance(x, list) else module(x)

        if self.main_route:
            out = self.make_ip(x, 2)
        else:
            out = self.make_ip(x, self.ip_scale_bands)
        
        out = apply(self.s1, out)
        out_c1 = self.c1(out)
        out = apply(self.s2, out_c1)
        out_c2 = self.c2(out)
        
        if self.bypass:
            bypass = self.s2b(out_c1)
            bypass = self.c2b_seq(bypass[0])
            bypass = bypass.reshape(bypass.size(0), -1)
        
        out = apply(self.s3, out_c2)
        out = self.global_pool(out)
        if isinstance(out, list):
            out = out[0]
        out = out.reshape(out.size(0), -1)

        if self.bypass:
            out = torch.cat([out, bypass], dim=1)
        
        out = self.fc(out)

        if self.contrastive_loss:
            if self.bypass:
                return out, out_c1, out_c2, bypass
            else:
                return out, out_c1, out_c2

        return out
    
    def print_param_stats(self):
        print(f"\nParameter breakdown for {self.__class__.__name__}:\n")
        total_params = 0
        stats = []
        for name, module in self.named_children():
            n_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
            total_params += n_params
            stats.append((name, n_params))

        stats.sort(key=lambda x: x[1], reverse=True)
        print(f"{'Module':30s} | {'# Params':>10s} | {'% of Total':>10s}")
        print("-" * 60)
        for name, count in stats:
            pct = 100 * count / total_params
            print(f"{name:30s} | {count:10,d} | {pct:10.2f}%")
        print("-" * 60)
        print(f"{'Total':30s} | {total_params:10,d} | {100.00:10.2f}%\n")

    
class RESMAX_V3_1(nn.Module):
    def __init__(self, num_classes=1000, in_chans=3, 
                 ip_scale_bands=1, classifier_input_size=512, contrastive_loss=False, pyramid=False,
                 bypass=False, main_route=False,
                 c_scoring='v2',
                 **kwargs):
        self.num_classes = num_classes
        self.in_chans = in_chans
        self.contrastive_loss = contrastive_loss
        self.ip_scale_bands = ip_scale_bands
        self.pyramid = pyramid
        self.bypass = bypass
        self.c_scoring = c_scoring
        self.main_route = main_route
        super(RESMAX_V3, self).__init__()

        self.s1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        # C1 using optimized layer
        self.c1 = C_scoring2_optimized(
            num_channels=64,
            pool_func1=nn.MaxPool2d(kernel_size=3, stride=2),
            pool_func2=nn.MaxPool2d(kernel_size=4, stride=3),
            skip=1,
            global_scale_pool=False
        )
        
        self.s2 = nn.Sequential(
            Residual1(64, 64),
            Residual1(64, 64)
        )

        # C2 using optimized layer
        self.c2 = C_scoring2_optimized(
            num_channels=64,
            pool_func1=nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            pool_func2=nn.MaxPool2d(kernel_size=6, stride=2),
            resize_kernel_1=3,
            resize_kernel_2=1,
            skip=2,
            global_scale_pool=False
        )
        
        if self.bypass:
            self.s2b = S2b_Res1()
            self.c2b_seq = nn.Sequential(
                nn.MaxPool2d(kernel_size=3, stride=2),
                nn.MaxPool2d(kernel_size=3, stride=2),
                nn.Conv2d(1024, 256, kernel_size=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True)
            )

        self.s3 = nn.Sequential(
            Residual1(64, 128, strides=2),
            Residual1(128, 128),
            Residual1(128, 256, strides=2),
            Residual1(256, 256),
            Residual1(256, 512, strides=2),
            Residual1(512, 512)
        )

        if self.ip_scale_bands > 4:
            self.global_pool = C_scoring2_optimized(
                num_channels=256,
                pool_func1=nn.MaxPool2d(kernel_size=3, stride=2),
                pool_func2=nn.MaxPool2d(kernel_size=6, stride=3, padding=1),
                resize_kernel_1=3,
                resize_kernel_2=1,
                skip=2,
                global_scale_pool=False
            )
        else:
            self.global_pool = C_adp(global_scale_pool=nn.AdaptiveAvgPool2d((1, 1)))

        self.fc= nn.Sequential(
            nn.Linear(classifier_input_size, num_classes)
        )

        self.print_param_stats()

    def make_ip(self, x, num_scale_bands):
        """
        Build an image pyramid.
        num_scale_bands = number of images in the pyramid - 1
        """
        base_image_size = int(x.shape[-1])
        scale_factor = 4  # exponent factor for scaling
        image_scales = get_ip_scales(num_scale_bands, base_image_size, scale_factor)
        
        if len(image_scales) > 1:
            image_pyramid = []
            for i_s in image_scales:
                i_s = int(i_s)
                interp_img = F.interpolate(x, size=(i_s, i_s), mode='bilinear', align_corners=False)
                image_pyramid.append(interp_img)
            return image_pyramid
        else:
            return [x]

    def forward(self, x, pyramid=False):
        def apply(module, x):
            return [module(xi) for xi in x] if isinstance(x, list) else module(x)

        if self.main_route:
            out = self.make_ip(x, 2)
        else:
            out = self.make_ip(x, self.ip_scale_bands)
        
        out = apply(self.s1, out)
        out_c1 = self.c1(out)
        out = apply(self.s2, out_c1)
        out_c2 = self.c2(out)
        
        if self.bypass:
            bypass = self.s2b(out_c1)
            bypass = self.c2b_seq(bypass[0])
            bypass = bypass.reshape(bypass.size(0), -1)
        
        out = apply(self.s3, out_c2)
        out = self.global_pool(out)
        if isinstance(out, list):
            out = out[0]
        out = out.reshape(out.size(0), -1)

        if self.bypass:
            out = torch.cat([out, bypass], dim=1)
        
        out = self.fc(out)

        if self.contrastive_loss:
            if self.bypass:
                return out, out_c1, out_c2, bypass
            else:
                return out, out_c1, out_c2

        return out
    
    def print_param_stats(self):
        print(f"\nParameter breakdown for {self.__class__.__name__}:\n")
        total_params = 0
        stats = []
        for name, module in self.named_children():
            n_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
            total_params += n_params
            stats.append((name, n_params))

        stats.sort(key=lambda x: x[1], reverse=True)
        print(f"{'Module':30s} | {'# Params':>10s} | {'% of Total':>10s}")
        print("-" * 60)
        for name, count in stats:
            pct = 100 * count / total_params
            print(f"{name:30s} | {count:10,d} | {pct:10.2f}%")
        print("-" * 60)
        print(f"{'Total':30s} | {total_params:10,d} | {100.00:10.2f}%\n")


class CHRESMAX_V3(nn.Module):
    """
    Example student-teacher style model with scale-consistency loss,
    using RESMAX_V2 as the backbone.

    In V3, the resmax_v2 returns full feature maps for C1 and C2 layers. Before
    the returned features are [0] for C1 and C2.
    """
    def __init__(self, 
                 num_classes=1000,
                 in_chans=3,
                 ip_scale_bands=1,
                 classifier_input_size=13312,
                 contrastive_loss=True,
                 bypass=False,
                 **kwargs):
        super().__init__()
        self.contrastive_loss = contrastive_loss
        self.num_classes = num_classes
        self.in_chans = in_chans
        self.ip_scale_bands = ip_scale_bands
        self.bypass = bypass
        
        # Use the optimized backbone
        self.model_backbone = RESMAX_V2(
            num_classes=num_classes,
            in_chans=in_chans,
            ip_scale_bands=self.ip_scale_bands,
            classifier_input_size=classifier_input_size,
            contrastive_loss=self.contrastive_loss,
            bypass=bypass,
        )

    def forward(self, x):
        """
        Creates two streams (original + random-scaled) for scale-consistency training.
        Returns:
            (output_of_stream1, correct_scale_loss)
        """
        # stream 1 (original scale)
        result = self.model_backbone(x)
        if self.bypass:
            stream_1_output, stream_1_c1_feats, stream_1_c2_feats, stream_1_bypass = result
        else:
            stream_1_output, stream_1_c1_feats, stream_1_c2_feats = result

        # stream 2 (random scale)
        scale_factor_list = [0.49, 0.59, 0.707, 0.841, 1.0, 1.189, 1.414, 1.681, 2.0]
        scale_factor = random.choice(scale_factor_list)
        img_hw = x.shape[-1]
        new_hw = int(img_hw * scale_factor)
        x_rescaled = F.interpolate(x, size=(new_hw, new_hw), mode='bilinear', align_corners=False)

        if new_hw <= img_hw:
            # pad if smaller
            x_rescaled = pad_to_size(x_rescaled, (img_hw, img_hw))
        else:
            # center-crop if bigger
            center_crop = transforms.CenterCrop(img_hw)
            x_rescaled = center_crop(x_rescaled)

        # forward pass on the scaled input
        result = self.model_backbone(x_rescaled)
        if self.bypass:
            stream_2_output, stream_2_c1_feats, stream_2_c2_feats, stream_2_bypass = result
        else:
            stream_2_output, stream_2_c1_feats, stream_2_c2_feats = result

        # Compute scale-consistency loss between the two streams, list ver
        c1_correct_scale_loss = 0
        for i in range(len(stream_1_c1_feats)):
            c1_correct_scale_loss += torch.mean(torch.abs(stream_1_c1_feats[i] - stream_2_c1_feats[i]))
        c1_correct_scale_loss /= len(stream_1_c1_feats)  # Average over all feature maps
        
        c2_correct_scale_loss = 0
        for i in range(len(stream_1_c2_feats)):
            c2_correct_scale_loss += torch.mean(torch.abs(stream_1_c2_feats[i] - stream_2_c2_feats[i]))
        c2_correct_scale_loss /= len(stream_1_c2_feats)  # Average over all feature maps
        
        out_correct_scale_loss = torch.mean(torch.abs(stream_1_output - stream_2_output))

        if self.bypass:
            bypass_correct_scale_loss = torch.mean(torch.abs(stream_1_bypass - stream_2_bypass))
        else:
            bypass_correct_scale_loss = 0

        correct_scale_loss = c1_correct_scale_loss + c2_correct_scale_loss + 0.1 * out_correct_scale_loss + bypass_correct_scale_loss

        return stream_1_output, correct_scale_loss

class CHRESMAX_V3_1(nn.Module):
    """
    Example student-teacher style model with scale-consistency loss,
    using RESMAX_V2 as the backbone.

    In V3, the resmax_v2 returns full feature maps for C1 and C2 layers. Before
    the returned features are [0] for C1 and C2.
    """
    def __init__(self, 
                 num_classes=1000,
                 in_chans=3,
                 ip_scale_bands=1,
                 classifier_input_size=13312,
                 contrastive_loss=True,
                 validation=True,   
                 bypass=False,
                 **kwargs):
        super().__init__()
        self.contrastive_loss = contrastive_loss
        self.num_classes = num_classes
        self.in_chans = in_chans
        self.ip_scale_bands = ip_scale_bands
        self.bypass = bypass
        self.validation = validation
        
        # Use the optimized backbone
        self.model_backbone = RESMAX_V2(
            num_classes=num_classes,
            in_chans=in_chans,
            ip_scale_bands=self.ip_scale_bands,
            classifier_input_size=classifier_input_size,
            contrastive_loss=self.contrastive_loss,
            bypass=bypass,
        )

    def forward(self, x):
        """
        Creates two streams (original + random-scaled) for scale-consistency training.
        Returns:
            (output_of_stream1, correct_scale_loss)
        """
        # stream 1 (original scale)
        #result = self.model_backbone(x)
        #if self.bypass:
        #    stream_1_output, stream_1_c1_feats, stream_1_c2_feats, stream_1_bypass = result
        #else:
        #    stream_1_output, stream_1_c1_feats, stream_1_c2_feats = result

        # stream 2 (random scale)
        scale_factor_list = [0.49, 0.59, 0.707, 0.841, 1.0, 1.189, 1.414, 1.681, 2.0]
        scale_factor = random.choice(scale_factor_list)
        img_hw = x.shape[-1]
        new_hw = int(img_hw * scale_factor)
        x_rescaled = F.interpolate(x, size=(new_hw, new_hw), mode='bilinear', align_corners=False)

        if new_hw <= img_hw:
            # pad if smaller
            x_rescaled = pad_to_size(x_rescaled, (img_hw, img_hw))
        else:
            # center-crop if bigger
            center_crop = transforms.CenterCrop(img_hw)
            x_rescaled = center_crop(x_rescaled)

        # forward pass on the scaled input
        result = self.model_backbone(x_rescaled)
        if self.bypass:
            stream_2_output, stream_2_c1_feats, stream_2_c2_feats, stream_2_bypass = result
        else:
            stream_2_output, stream_2_c1_feats, stream_2_c2_feats = result

        if self.validation:
            return stream_2_output

        # Compute scale-consistency loss between the two streams, list ver
        c1_correct_scale_loss = 0
        for i in range(len(stream_1_c1_feats)):
            c1_correct_scale_loss += torch.mean(torch.abs(stream_1_c1_feats[i] - stream_2_c1_feats[i]))
        c1_correct_scale_loss /= len(stream_1_c1_feats)  # Average over all feature maps
        
        c2_correct_scale_loss = 0
        for i in range(len(stream_1_c2_feats)):
            c2_correct_scale_loss += torch.mean(torch.abs(stream_1_c2_feats[i] - stream_2_c2_feats[i]))
        c2_correct_scale_loss /= len(stream_1_c2_feats)  # Average over all feature maps
        
        out_correct_scale_loss = torch.mean(torch.abs(stream_1_output - stream_2_output))

        if self.bypass:
            bypass_correct_scale_loss = torch.mean(torch.abs(stream_1_bypass - stream_2_bypass))
        else:
            bypass_correct_scale_loss = 0

        correct_scale_loss = c1_correct_scale_loss + c2_correct_scale_loss + 0.1 * out_correct_scale_loss + bypass_correct_scale_loss

        return stream_1_output, correct_scale_loss
    
def rescaling_x_to_scale_band(x, num_scale_bands, scale_band, center):
    """
    Rescale the input x to the scale band centered at center.
    Distributed training compatible version with explicit error handling.
    """
    try:
        # Ensure inputs are tensors and on the same device
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x)
        if not isinstance(center, torch.Tensor):
            center = torch.as_tensor(center)
        
        # Get device information
        device = x.device
        image_sizes = get_ip_scales(num_scale_bands, x.shape[-1])
        img_hw = x.shape[-1]
        
        # Initialize output tensor
        batch_size = len(scale_band)
        output = torch.empty((batch_size, x.shape[1], img_hw, img_hw), 
                           dtype=x.dtype, device=device)
        
        # Process each image
        for i in range(batch_size):
            # Get current image and center
            current_img = x[i:i+1]
            current_center = center[i]
            
            # Determine target size
            if scale_band[i] > len(image_sizes):
                new_size = int(image_sizes[-1])
            else:
                new_size = int(image_sizes[scale_band[i]-1])
            
            # Get center coordinates
            center_x, center_y = current_center[0].item(), current_center[1].item()
            
            # Calculate crop size based on target size
            # For zoom in (new_size < img_hw), we want a smaller crop
            # For zoom out (new_size > img_hw), we want a larger crop
            crop_size = int(img_hw * (img_hw / new_size))
            
            # Calculate crop boundaries
            start_x = max(0, int(center_x - crop_size // 2))
            start_y = max(0, int(center_y - crop_size // 2))
            end_x = min(current_img.shape[-1], start_x + crop_size)
            end_y = min(current_img.shape[-2], start_y + crop_size)
            
            # Adjust boundaries if needed
            if end_x - start_x < crop_size:
                start_x = max(0, end_x - crop_size)
            if end_y - start_y < crop_size:
                start_y = max(0, end_y - crop_size)
            
            # Crop and resize
            cropped = current_img[..., start_y:end_y, start_x:end_x]
            
            # Resize to target size
            resized = F.interpolate(cropped, size=(new_size, new_size), mode='bilinear', align_corners=False)
            
            # Pad or crop to final size
            if new_size <= img_hw:
                # Pad if smaller (zoom out)
                resized = pad_to_size(resized, (img_hw, img_hw))
            else:
                # Center crop if bigger (zoom in)
                center_crop = transforms.CenterCrop(img_hw)
                resized = center_crop(resized)
            
            # Store result
            output[i] = resized[0]
        
        return output
        
    except Exception as e:
        # Log the error and re-raise
        logging.error(f"Error in rescaling_x_to_scale_band: {str(e)}")
        raise


class CHRESMAX_V3_S(nn.Module):
    """
    Example student-teacher style model with scale-consistency loss,
    using RESMAX_V2 as the backbone.

    In V3, the resmax_v2 returns full feature maps for C1 and C2 layers. Before
    the returned features are [0] for C1 and C2.
    """
    def __init__(self, 
                 num_classes=1000,
                 in_chans=3,
                 ip_scale_bands=1,
                 classifier_input_size=13312,
                 contrastive_loss=True,
                 bypass=False,debug=False,
                 **kwargs):
        super().__init__()
        self.contrastive_loss = contrastive_loss
        self.num_classes = num_classes
        self.in_chans = in_chans
        self.ip_scale_bands = ip_scale_bands
        self.bypass = bypass
        self.debug = debug
        # Use the optimized backbone
        self.model_backbone = RESMAX_V2(
            num_classes=num_classes,
            in_chans=in_chans,
            ip_scale_bands=self.ip_scale_bands,
            classifier_input_size=classifier_input_size,
            contrastive_loss=self.contrastive_loss,
            bypass=bypass,
        )

    def forward(self, x, scale_band=None, center=None):
        """
        Creates two streams (original + random-scaled) for scale-consistency training.
        Returns:
            (output_of_stream1, correct_scale_loss)
        """
        # stream 1 (original scale)
        result = self.model_backbone(x,main_route=True)
        if self.bypass:
            stream_1_output, stream_1_c1_feats, stream_1_c2_feats, stream_1_bypass = result
        else:
            stream_1_output, stream_1_c1_feats, stream_1_c2_feats = result
        
        if scale_band is not None and center is not None: 
            # Save rescaled images
            x_rescaled = rescaling_x_to_scale_band(x, self.ip_scale_bands, scale_band, center)
            if self.debug:
                save_dir = 'debugging_images'
                os.makedirs(save_dir, exist_ok=True)
                for i in range(len(x)):
                    save_path = os.path.join(save_dir, f'original_image_{i}.png')
                    save_image(x[i], save_path)
                for i, img in enumerate(x_rescaled):
                    save_path = os.path.join(save_dir, f'rescaled_image_{i}.png')
                    save_image(img, save_path)
        else:
            x_rescaled = x
        
        # forward pass on the scaled input
        result = self.model_backbone(x_rescaled,main_route=True)
        if self.bypass:
            stream_2_output, stream_2_c1_feats, stream_2_c2_feats, stream_2_bypass = result
        else:
            stream_2_output, stream_2_c1_feats, stream_2_c2_feats = result

        # Compute scale-consistency loss between the two streams, list ver
        c1_correct_scale_loss = 0
        for i in range(len(stream_1_c1_feats)):
            c1_correct_scale_loss += torch.mean(torch.abs(stream_1_c1_feats[i] - stream_2_c1_feats[i]))
        c1_correct_scale_loss /= len(stream_1_c1_feats)  # Average over all feature maps
        
        c2_correct_scale_loss = 0
        for i in range(len(stream_1_c2_feats)):
            c2_correct_scale_loss += torch.mean(torch.abs(stream_1_c2_feats[i] - stream_2_c2_feats[i]))
        c2_correct_scale_loss /= len(stream_1_c2_feats)  # Average over all feature maps
        
        out_correct_scale_loss = torch.mean(torch.abs(stream_1_output - stream_2_output))

        if self.bypass:
            bypass_correct_scale_loss = torch.mean(torch.abs(stream_1_bypass - stream_2_bypass))
        else:
            bypass_correct_scale_loss = 0

        correct_scale_loss = c1_correct_scale_loss + c2_correct_scale_loss + 0.1 * out_correct_scale_loss + bypass_correct_scale_loss

        return stream_1_output, correct_scale_loss

class CHRESMAX_V4(nn.Module):
    """
    Example student-teacher style model with scale-consistency loss,
    using RESMAX_V2 as the backbone.

    In V4, we use new loss clip loss. also use reflect for padding
    In V3, the resmax_v2 returns full feature maps for C1 and C2 layers. Before
    the returned features are [0] for C1 and C2.
    """
    def __init__(self, 
                 num_classes=1000,
                 in_chans=3,
                 ip_scale_bands=1,
                 classifier_input_size=13312,
                 contrastive_loss=True,
                 bypass=False,
                 temperature=0.1,
                 **kwargs):
        super().__init__()
        self.contrastive_loss = contrastive_loss
        self.num_classes = num_classes
        self.in_chans = in_chans
        self.ip_scale_bands = ip_scale_bands
        self.bypass = bypass
        self.temperature = temperature
        
        # Use the optimized backbone
        self.backbone = RESMAX_V2(
            num_classes=num_classes,
            in_chans=in_chans,
            ip_scale_bands=self.ip_scale_bands,
            classifier_input_size=classifier_input_size,
            contrastive_loss=self.contrastive_loss,
            bypass=bypass,
        )

    def forward(self, x):
        batch_size = x.shape[0]
        
        # Original input - full forward pass
        if self.backbone.bypass:
            out1, c1_feats1, c2_feats1, bypass1 = self.backbone(x)
        else:
            out1, c1_feats1, c2_feats1 = self.backbone(x)
        
        # Randomly scaled input
        scale_factor_list = [0.49, 0.59, 0.707, 0.841, 1.0, 1.189, 1.414, 1.681, 2.0]
        scale_factor = random.choice(scale_factor_list)
        img_hw = x.shape[-1]
        new_hw = int(img_hw * scale_factor)
        x_rescaled = F.interpolate(x, size=(new_hw, new_hw), mode='bilinear', align_corners=False)

        if new_hw <= img_hw:
            # pad if smaller
            # 'constant', 'reflect', 'replicate' or 'circular'. Default: 'constant'
            x_rescaled = pad_to_size(x_rescaled, (img_hw, img_hw), mode='reflect')
        else:
            # center-crop if bigger
            center_crop = transforms.CenterCrop(img_hw)
            x_rescaled = center_crop(x_rescaled)
            
        # Forward pass on scaled input
        if self.backbone.bypass:
            out2, c1_feats2, c2_feats2, bypass2 = self.backbone(x_rescaled)
        else:
            out2, c1_feats2, c2_feats2 = self.backbone(x_rescaled)


        def clip_style_loss(z1, z2, temperature=self.temperature):
            """
            CLIP-style contrastive loss between original and scaled features.
            - z1: features from original images [B, D]
            - z2: features from rescaled images [B, D]
            """
            # Normalize
            z1 = F.normalize(z1, dim=1)  # [B, D]
            z2 = F.normalize(z2, dim=1)  # [B, D]

            # Compute logits
            logits_per_orig = torch.matmul(z1, z2.T) / temperature  # [B, B]
            logits_per_scaled = torch.matmul(z2, z1.T) / temperature  # [B, B]

            # Labels are indices [0, 1, ..., B-1]
            labels = torch.arange(z1.size(0), device=z1.device)

            loss_orig = F.cross_entropy(logits_per_orig, labels)
            loss_scaled = F.cross_entropy(logits_per_scaled, labels)

            return (loss_orig + loss_scaled) / 2
        
        # Compute CLIP-style contrastive loss between features
        if isinstance(c1_feats1, list) and isinstance(c1_feats2, list):
            c1_contrastive_loss = 0
            for i in range(len(c1_feats1)):
                f1 = c1_feats1[i].reshape(c1_feats1[i].size(0), -1)
                f2 = c1_feats2[i].reshape(c1_feats2[i].size(0), -1)
                c1_contrastive_loss += clip_style_loss(f1, f2)
            c1_contrastive_loss /= len(c1_feats1)
        else:
            f1 = c1_feats1.reshape(c1_feats1.size(0), -1)
            f2 = c1_feats2.reshape(c2_feats2.size(0), -1)
            c1_contrastive_loss = clip_style_loss(f1, f2)

        if isinstance(c2_feats1, list) and isinstance(c2_feats2, list):
            c2_contrastive_loss = 0
            for i in range(len(c2_feats1)):
                f1 = c2_feats1[i].reshape(c2_feats1[i].size(0), -1)
                f2 = c2_feats2[i].reshape(c2_feats2[i].size(0), -1)
                c2_contrastive_loss += clip_style_loss(f1, f2)
            c2_contrastive_loss /= len(c2_feats1)
        else:
            f1 = c2_feats1.reshape(c2_feats1.size(0), -1)
            f2 = c2_feats2.reshape(c2_feats2.size(0), -1)
            c2_contrastive_loss = clip_style_loss(f1, f2)

        # Top-level output loss
        out_contrastive_loss = clip_style_loss(out1, out2)

        # Optional: bypass contrastive
        if self.backbone.bypass:
            bypass_contrastive_loss = clip_style_loss(bypass1.reshape(bypass1.size(0), -1), bypass2.reshape(bypass2.size(0), -1))
            total_loss = c1_contrastive_loss + c2_contrastive_loss + 0.1 * out_contrastive_loss + 0.1 * bypass_contrastive_loss
        else:
            total_loss = c1_contrastive_loss + c2_contrastive_loss + 0.1 * out_contrastive_loss

                            
        return out1, total_loss
    
class CHRESMAX_V5(nn.Module):
    """
    Example student-teacher style model with scale-consistency loss,
    using RESMAX_V2 as the backbone.

    In V4, we use new loss clip loss. also use reflect for padding
    In V3, the resmax_v2 returns full feature maps for C1 and C2 layers. Before
    the returned features are [0] for C1 and C2.
    """
    def __init__(self, 
                 num_classes=1000,
                 in_chans=3,
                 ip_scale_bands=1,
                 classifier_input_size=512,
                 contrastive_loss=True,
                 bypass=False,
                 temperature=0.1,
                 **kwargs):
        super().__init__()
        self.contrastive_loss = contrastive_loss
        self.num_classes = num_classes
        self.in_chans = in_chans
        self.ip_scale_bands = ip_scale_bands
        self.bypass = bypass
        self.temperature = temperature
        
        # Use the optimized backbone
        self.backbone = RESMAX_V3(
            num_classes=num_classes,
            in_chans=in_chans,
            ip_scale_bands=self.ip_scale_bands,
            classifier_input_size=classifier_input_size,
            contrastive_loss=self.contrastive_loss,
            bypass=bypass,
        )

    def forward(self, x):
        batch_size = x.shape[0]
        
        # Original input - full forward pass
        if self.backbone.bypass:
            out1, c1_feats1, c2_feats1, bypass1 = self.backbone(x)
        else:
            out1, c1_feats1, c2_feats1 = self.backbone(x)
        
        # Randomly scaled input
        scale_factor_list = [0.49, 0.59, 0.707, 0.841, 1.0, 1.189, 1.414, 1.681, 2.0]
        scale_factor = random.choice(scale_factor_list)
        img_hw = x.shape[-1]
        new_hw = int(img_hw * scale_factor)
        x_rescaled = F.interpolate(x, size=(new_hw, new_hw), mode='bilinear', align_corners=False)

        if new_hw <= img_hw:
            # pad if smaller
            # 'constant', 'reflect', 'replicate' or 'circular'. Default: 'constant'
            x_rescaled = pad_to_size(x_rescaled, (img_hw, img_hw), mode='reflect')
        else:
            # center-crop if bigger
            center_crop = transforms.CenterCrop(img_hw)
            x_rescaled = center_crop(x_rescaled)
            
        # Forward pass on scaled input
        if self.backbone.bypass:
            out2, c1_feats2, c2_feats2, bypass2 = self.backbone(x_rescaled)
        else:
            out2, c1_feats2, c2_feats2 = self.backbone(x_rescaled)


        def clip_style_loss(z1, z2, temperature=self.temperature):
            """
            CLIP-style contrastive loss between original and scaled features.
            - z1: features from original images [B, D]
            - z2: features from rescaled images [B, D]
            """
            # Normalize
            z1 = F.normalize(z1, dim=1)  # [B, D]
            z2 = F.normalize(z2, dim=1)  # [B, D]

            # Compute logits
            logits_per_orig = torch.matmul(z1, z2.T) / temperature  # [B, B]
            logits_per_scaled = torch.matmul(z2, z1.T) / temperature  # [B, B]

            # Labels are indices [0, 1, ..., B-1]
            labels = torch.arange(z1.size(0), device=z1.device)

            loss_orig = F.cross_entropy(logits_per_orig, labels)
            loss_scaled = F.cross_entropy(logits_per_scaled, labels)

            return (loss_orig + loss_scaled) / 2
        
        # Compute CLIP-style contrastive loss between features
        if isinstance(c1_feats1, list) and isinstance(c1_feats2, list):
            c1_contrastive_loss = 0
            for i in range(len(c1_feats1)):
                f1 = c1_feats1[i].reshape(c1_feats1[i].size(0), -1)
                f2 = c1_feats2[i].reshape(c1_feats2[i].size(0), -1)
                c1_contrastive_loss += clip_style_loss(f1, f2)
            c1_contrastive_loss /= len(c1_feats1)
        else:
            f1 = c1_feats1.reshape(c1_feats1.size(0), -1)
            f2 = c1_feats2.reshape(c2_feats2.size(0), -1)
            c1_contrastive_loss = clip_style_loss(f1, f2)

        if isinstance(c2_feats1, list) and isinstance(c2_feats2, list):
            c2_contrastive_loss = 0
            for i in range(len(c2_feats1)):
                f1 = c2_feats1[i].reshape(c2_feats1[i].size(0), -1)
                f2 = c2_feats2[i].reshape(c2_feats2[i].size(0), -1)
                c2_contrastive_loss += clip_style_loss(f1, f2)
            c2_contrastive_loss /= len(c2_feats1)
        else:
            f1 = c2_feats1.reshape(c2_feats1.size(0), -1)
            f2 = c2_feats2.reshape(c2_feats2.size(0), -1)
            c2_contrastive_loss = clip_style_loss(f1, f2)

        # Top-level output loss
        out_contrastive_loss = clip_style_loss(out1, out2)

        # Optional: bypass contrastive
        if self.backbone.bypass:
            bypass_contrastive_loss = clip_style_loss(bypass1.reshape(bypass1.size(0), -1), bypass2.reshape(bypass2.size(0), -1))
            total_loss = c1_contrastive_loss + c2_contrastive_loss + 0.1 * out_contrastive_loss + 0.1 * bypass_contrastive_loss
        else:
            total_loss = c1_contrastive_loss + c2_contrastive_loss + 0.1 * out_contrastive_loss


        return out1, total_loss

# Create a new model for contrastive fine-tuning
class ContrastiveRESMAX(nn.Module):
    def __init__(self,
                num_classes=1000,
                in_chans=3,
                ip_scale_bands=1,
                classifier_input_size=9216,
                contrastive_loss=True,
                bypass=False,
                pretrained_path=None,
                temperature=0.1,
                **kwargs):
        super().__init__()

        self.num_classes = num_classes
        self.contrastive_loss = contrastive_loss

        pretrained_path = f'/oscar/data/tserre/xyu110/pytorch-output/train/recent_results/ip_{ip_scale_bands}_resmax_v2_gpu_8_cl_0_ip_3_322_322_{classifier_input_size}_c1[_6,3,1_]/model_best.pth.tar'
        self.temperature = temperature
        self.ip_scale_bands = ip_scale_bands

        # Create the backbone model
        self.backbone = RESMAX_V2(
            num_classes=num_classes,
            in_chans=in_chans,
            ip_scale_bands=self.ip_scale_bands,
            classifier_input_size=classifier_input_size,
            contrastive_loss=self.contrastive_loss,
            bypass=bypass,
        )
        
        # Load pretrained weights from file path
        checkpoint = torch.load(pretrained_path, weights_only=False, map_location='cpu')
        
        # Handle different checkpoint formats
        if 'state_dict' in checkpoint:
            # If checkpoint contains a state_dict key (common in training frameworks)
            self.backbone.load_state_dict(checkpoint['state_dict'], strict=True)
            print('Loaded state dict from checkpoint if')
        else:
            # Directly load if it's just the state dict
            self.backbone.load_state_dict(checkpoint, strict=True)
            print('Loaded state dict from checkpoint else')
        
        # Freeze backbone parameters
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        # Unfreeze layers strategically for contrastive fine-tuning
        
        # Always unfreeze the FC layers
        for param in self.backbone.fc2.parameters():
            param.requires_grad = True
        for param in self.backbone.fc1.parameters():
            param.requires_grad = True
        for param in self.backbone.fc.parameters():
            param.requires_grad = True
            
        # # Unfreeze S3_Res (deeper convolutional layers)
        # # This is crucial since S3_Res processes features after C2 scoring,
        # # which directly impacts scale robustness
        for param in self.backbone.s3.parameters():
            param.requires_grad = True
        # for param in self.backbone.c1.parameters():
        #     param.requires_grad = True
        # for param in self.backbone.c2.parameters():
        #     param.requires_grad = True
        # if bypass:
        #     for param in self.backbone.c2b_seq.parameters():
        #         param.requires_grad = True
            
        # Unfreeze global pooling layer that aggregates features
        for param in self.backbone.global_pool.parameters():
            param.requires_grad = True
            
        # No additional projection head needed
        # We'll use the backbone's FC layers for feature extraction
    
    def forward(self, x):
        batch_size = x.shape[0]
        
        # Original input - full forward pass
        if self.backbone.bypass:
            out1, c1_feats1, c2_feats1, bypass1 = self.backbone(x)
        else:
            out1, c1_feats1, c2_feats1 = self.backbone(x)
        
        # Randomly scaled input
        scale_factor_list = [0.49, 0.59, 0.707, 0.841, 1.0, 1.189, 1.414, 1.681, 2.0]
        scale_factor = random.choice(scale_factor_list)
        img_hw = x.shape[-1]
        new_hw = int(img_hw * scale_factor)
        x_rescaled = F.interpolate(x, size=(new_hw, new_hw), mode='bilinear', align_corners=False)

        if new_hw <= img_hw:
            # pad if smaller
            x_rescaled = pad_to_size(x_rescaled, (img_hw, img_hw))
        else:
            # center-crop if bigger
            center_crop = transforms.CenterCrop(img_hw)
            x_rescaled = center_crop(x_rescaled)
            
        # Forward pass on scaled input
        if self.backbone.bypass:
            out2, c1_feats2, c2_feats2, bypass2 = self.backbone(x_rescaled)
        else:
            out2, c1_feats2, c2_feats2 = self.backbone(x_rescaled)

        ##############################NT-Xent Loss#############################
        
        # Helper function to compute NT-Xent loss between two feature maps
        def nt_xent_loss(f1, f2, temperature=self.temperature):
            # Flatten spatial dimensions if needed
            if len(f1.shape) > 2:
                f1 = f1.reshape(f1.size(0), -1)
                f2 = f2.reshape(f2.size(0), -1)
            
            # Normalize features
            z1 = F.normalize(f1, dim=1)
            z2 = F.normalize(f2, dim=1)
            
            # Concatenate features from both scales
            features = torch.cat([z1, z2], dim=0)
            
            # Compute similarity matrix
            sim_matrix = torch.matmul(features, features.T) / temperature
            
            # Create mask for positive pairs
            pos_mask = torch.zeros_like(sim_matrix)
            pos_mask[:batch_size, batch_size:] = torch.eye(batch_size)
            pos_mask[batch_size:, :batch_size] = torch.eye(batch_size)
            
            # Create mask to exclude self-similarity
            self_mask = torch.eye(2 * batch_size, device=sim_matrix.device)
            logits_mask = torch.ones_like(sim_matrix) - self_mask
            
            # NT-Xent loss calculation
            exp_logits = torch.exp(sim_matrix) * logits_mask
            log_prob = sim_matrix - torch.log(exp_logits.sum(dim=1, keepdim=True))
            mean_log_prob_pos = (pos_mask * log_prob).sum(1) / pos_mask.sum(1)
            
            return -mean_log_prob_pos.mean()
        
        # Compute NT-Xent loss between features from different scales
        if isinstance(c1_feats1, list) and isinstance(c1_feats2, list):
            # Handle list of feature maps
            c1_contrastive_loss = 0
            for i in range(len(c1_feats1)):
                c1_contrastive_loss += nt_xent_loss(c1_feats1[i], c1_feats2[i])
            c1_contrastive_loss /= len(c1_feats1)  # Average loss across feature maps
        else:
            # Direct feature map
            c1_contrastive_loss = nt_xent_loss(c1_feats1, c1_feats2)

        if isinstance(c2_feats1, list) and isinstance(c2_feats2, list):
            # Handle list of feature maps
            c2_contrastive_loss = 0
            for i in range(len(c2_feats1)):
                c2_contrastive_loss += nt_xent_loss(c2_feats1[i], c2_feats2[i])
            c2_contrastive_loss /= len(c2_feats1)  # Average loss across feature maps
        else:
            # Direct feature map
            c2_contrastive_loss = nt_xent_loss(c2_feats1, c2_feats2)

        # Compute output contrastive loss (this should be a single tensor, not a list)
        out_contrastive_loss = nt_xent_loss(out1, out2)

        # Compute bypass contrastive loss if needed
        if self.backbone.bypass:
            bypass_contrastive_loss = nt_xent_loss(bypass1, bypass2)
            total_loss = c1_contrastive_loss + c2_contrastive_loss + 0.1 * out_contrastive_loss + 0.1 * bypass_contrastive_loss
        else:
            total_loss = c1_contrastive_loss + c2_contrastive_loss + 0.1 * out_contrastive_loss 

        # print(f"c1_contrastive_loss: {c1_contrastive_loss}, c2_contrastive_loss: {c2_contrastive_loss}, out_contrastive_loss: {out_contrastive_loss}")
                
        return out1, total_loss
    

# Create a new model for contrastive fine-tuning
class ContrastiveRESMAXV1(nn.Module):
    def __init__(self,
                num_classes=1000,
                in_chans=3,
                ip_scale_bands=1,
                classifier_input_size=9216,
                contrastive_loss=True,
                bypass=False,
                pretrained_path=None,
                temperature=0.1,
                **kwargs):
        super().__init__()

        self.num_classes = num_classes
        self.contrastive_loss = contrastive_loss

        pretrained_path = f'/oscar/data/tserre/xyu110/pytorch-output/train/recent_results/ip_{ip_scale_bands}_resmax_v2_gpu_8_cl_0_ip_3_322_322_{classifier_input_size}_c1[_6,3,1_]/model_best.pth.tar'
        self.temperature = temperature
        self.ip_scale_bands = ip_scale_bands

        # Create the backbone model
        self.backbone = RESMAX_V2(
            num_classes=num_classes,
            in_chans=in_chans,
            ip_scale_bands=self.ip_scale_bands,
            classifier_input_size=classifier_input_size,
            contrastive_loss=self.contrastive_loss,
            bypass=bypass,
        )
        
        # Load pretrained weights from file path
        checkpoint = torch.load(pretrained_path, weights_only=False, map_location='cpu')
        
        # Handle different checkpoint formats
        if 'state_dict' in checkpoint:
            # If checkpoint contains a state_dict key (common in training frameworks)
            self.backbone.load_state_dict(checkpoint['state_dict'], strict=True)
            print('Loaded state dict from checkpoint if')
        else:
            # Directly load if it's just the state dict
            self.backbone.load_state_dict(checkpoint, strict=True)
            print('Loaded state dict from checkpoint else')
        
        # Freeze backbone parameters
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        # Unfreeze layers strategically for contrastive fine-tuning
        
        # Always unfreeze the FC layers
        for param in self.backbone.fc2.parameters():
            param.requires_grad = True
        for param in self.backbone.fc1.parameters():
            param.requires_grad = True
        for param in self.backbone.fc.parameters():
            param.requires_grad = True
            
        # # Unfreeze S3_Res (deeper convolutional layers)
        for param in self.backbone.s3.parameters():
            param.requires_grad = True

        # Unfreeze global pooling layer that aggregates features
        for param in self.backbone.global_pool.parameters():
            param.requires_grad = True
            
        # No additional projection head needed
        # We'll use the backbone's FC layers for feature extraction
    
    def forward(self, x):
        batch_size = x.shape[0]
        
        # Original input - full forward pass
        if self.backbone.bypass:
            out1, c1_feats1, c2_feats1, bypass1 = self.backbone(x)
        else:
            out1, c1_feats1, c2_feats1 = self.backbone(x)
        
        # Randomly scaled input
        scale_factor_list = [0.49, 0.59, 0.707, 0.841, 1.0, 1.189, 1.414, 1.681, 2.0]
        scale_factor = random.choice(scale_factor_list)
        img_hw = x.shape[-1]
        new_hw = int(img_hw * scale_factor)
        x_rescaled = F.interpolate(x, size=(new_hw, new_hw), mode='bilinear', align_corners=False)

        if new_hw <= img_hw:
            # pad if smaller
            # 'constant', 'reflect', 'replicate' or 'circular'. Default: 'constant'
            x_rescaled = pad_to_size(x_rescaled, (img_hw, img_hw), mode='reflect')
        else:
            # center-crop if bigger
            center_crop = transforms.CenterCrop(img_hw)
            x_rescaled = center_crop(x_rescaled)
            
        # Forward pass on scaled input
        if self.backbone.bypass:
            out2, c1_feats2, c2_feats2, bypass2 = self.backbone(x_rescaled)
        else:
            out2, c1_feats2, c2_feats2 = self.backbone(x_rescaled)

        ##############################NT-Xent Loss#############################
        
        # Helper function to compute NT-Xent loss between two feature maps
        def clip_style_loss(z1, z2, temperature=self.temperature):
            """
            CLIP-style contrastive loss between original and scaled features.
            - z1: features from original images [B, D]
            - z2: features from rescaled images [B, D]
            """
            # Normalize
            z1 = F.normalize(z1, dim=1)  # [B, D]
            z2 = F.normalize(z2, dim=1)  # [B, D]

            # Compute logits
            logits_per_orig = torch.matmul(z1, z2.T) / temperature  # [B, B]
            logits_per_scaled = torch.matmul(z2, z1.T) / temperature  # [B, B]

            # Labels are indices [0, 1, ..., B-1]
            labels = torch.arange(z1.size(0), device=z1.device)

            loss_orig = F.cross_entropy(logits_per_orig, labels)
            loss_scaled = F.cross_entropy(logits_per_scaled, labels)

            return (loss_orig + loss_scaled) / 2
        
        # Compute CLIP-style contrastive loss between features
        if isinstance(c1_feats1, list) and isinstance(c1_feats2, list):
            c1_contrastive_loss = 0
            for i in range(len(c1_feats1)):
                f1 = c1_feats1[i].reshape(c1_feats1[i].size(0), -1)
                f2 = c1_feats2[i].reshape(c1_feats2[i].size(0), -1)
                c1_contrastive_loss += clip_style_loss(f1, f2)
            c1_contrastive_loss /= len(c1_feats1)
        else:
            f1 = c1_feats1.reshape(c1_feats1.size(0), -1)
            f2 = c1_feats2.reshape(c2_feats2.size(0), -1)
            c1_contrastive_loss = clip_style_loss(f1, f2)

        if isinstance(c2_feats1, list) and isinstance(c2_feats2, list):
            c2_contrastive_loss = 0
            for i in range(len(c2_feats1)):
                f1 = c2_feats1[i].reshape(c2_feats1[i].size(0), -1)
                f2 = c2_feats2[i].reshape(c2_feats2[i].size(0), -1)
                c2_contrastive_loss += clip_style_loss(f1, f2)
            c2_contrastive_loss /= len(c2_feats1)
        else:
            f1 = c2_feats1.reshape(c2_feats1.size(0), -1)
            f2 = c2_feats2.reshape(c2_feats2.size(0), -1)
            c2_contrastive_loss = clip_style_loss(f1, f2)

        # Top-level output loss
        out_contrastive_loss = clip_style_loss(out1, out2)

        # Optional: bypass contrastive
        if self.backbone.bypass:
            bypass_contrastive_loss = clip_style_loss(bypass1.reshape(bypass1.size(0), -1), bypass2.reshape(bypass2.size(0), -1))
            total_loss = c1_contrastive_loss + c2_contrastive_loss + 0.1 * out_contrastive_loss + 0.1 * bypass_contrastive_loss
        else:
            total_loss = c1_contrastive_loss + c2_contrastive_loss + 0.1 * out_contrastive_loss

                            
        return out1, total_loss
class RESMAX_Bypass(nn.Module):
    def __init__(self, num_classes=1000, big_size=322, small_size=227, in_chans=3, 
                 ip_scale_bands=1, classifier_input_size=13312, contrastive_loss=False, pyramid=False, **kwargs):
        self.num_classes = num_classes
        self.in_chans = in_chans
        self.contrastive_loss = contrastive_loss
        self.ip_scale_bands = ip_scale_bands
        self.pyramid = pyramid
        self.big_size = big_size
        self.small_size = small_size
        super(RESMAX_Bypass, self).__init__()

        self.s1_big = S1_Res_Big()
        self.s1_small = S1_Res_Small()
        
        self.c1 = C_scoring(96,
            nn.MaxPool2d(kernel_size=6, stride=3, padding=3),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        
        self.s2 = S2_Res()
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

    def forward(self, x, pyramid=False):
        # resize image
        # always making pyramid to start with only two scales. 
        out = self.make_ip(x)
        
        ## should make SxBxCxHxW
        out_1 = self.s1_big([out[0]])
        out_2 = self.s1_small([out[1]])
        out_c1 = self.c1([out_1[0],out_2[0]])
        
        # s2
        out = self.s2(out_c1)
        out_c2 = self.c2(out)
        
        # s2b # input torch.Size([128, 96, 28, 28])
        bypass = self.s2b(out_c1) # torch.Size([128, 1024, 28, 28])
        bypass = self.c2b_seq(bypass[0]) # torch.Size([128, 256, 6, 6])
        bypass = bypass.reshape(bypass.size(0), -1)

        # s3
        out = self.s3(out_c2)
        out = self.global_pool(out) # out shape: torch.Size([128, 256, 6, 6])
        out = out.reshape(out.size(0), -1)

        # merge bypass and main path
        out = torch.cat([out, bypass], dim=1)
        # out += bypass

        # fc layers
        out = self.fc(out)
        out = self.fc1(out)
        out = self.fc2(out)

        if self.contrastive_loss:
            return out, out_c1[0],out_c2[0]

        return out
    
@register_model
def resmax_v2(pretrained=False, **kwargs):
    #deleting some kwargs that are messing up training
    try:
        del kwargs["pretrained_cfg"]
        del kwargs["pretrained_cfg_overlay"]
        del kwargs["drop_rate"]
    except:
        pass
    model = RESMAX_V2(**kwargs)

    return model

@register_model
def resmax_v3(pretrained=False, **kwargs):
    #deleting some kwargs that are messing up training
    try:
        del kwargs["pretrained_cfg"]
        del kwargs["pretrained_cfg_overlay"]
        del kwargs["drop_rate"]
    except:
        pass
    model = RESMAX_V3(**kwargs)


@register_model
def chresmax_v3(pretrained=False, **kwargs):
    """
    Registry function to create a CHALEXMAX_V3_3_optimized model
    via timm's create_model API.
    """
    for key in ["pretrained_cfg", "pretrained_cfg_overlay", "drop_rate"]:
        kwargs.pop(key, None)

    for key, val in kwargs.items():
        print(key, val)

    model = CHRESMAX_V3(**kwargs)
    return model

@register_model
def chresmax_v4(pretrained=False, **kwargs):
    """
    Registry function to create a CHALEXMAX_V3_3_optimized model
    via timm's create_model API.
    """
    for key in ["pretrained_cfg", "pretrained_cfg_overlay", "drop_rate"]:
        kwargs.pop(key, None)

    for key, val in kwargs.items():
        print(key, val)

    model = CHRESMAX_V4(**kwargs)
    return model

@register_model
def chresmax_v3_1(pretrained=False, **kwargs):
    """
    Registry function to create a CHRESMAX_V3_1 model
    via timm's create_model API.
    """
    for key in ["pretrained_cfg", "pretrained_cfg_overlay", "drop_rate"]:
        kwargs.pop(key, None)

    for key, val in kwargs.items():
        print(key, val)

    model = CHRESMAX_V3_1(**kwargs)
    return model

@register_model
def chresmax_v5(pretrained=False, **kwargs):
    """
    Registry function to create a CHALEXMAX_V3_3_optimized model
    via timm's create_model API.
    """
    for key in ["pretrained_cfg", "pretrained_cfg_overlay", "drop_rate"]:
        kwargs.pop(key, None)

    for key, val in kwargs.items():
        print(key, val)

    model = CHRESMAX_V5(**kwargs)
    return model

@register_model
def contrastive_resmax(pretrained=False, **kwargs):
    """
    Registry function to create a ContrastiveRESMAX model
    via timm's create_model API.
    """
    for key in ["pretrained_cfg", "pretrained_cfg_overlay", "drop_rate"]:
        kwargs.pop(key, None)

    for key, val in kwargs.items():
        print(key, val)

    if pretrained:
        pass
    
    model = ContrastiveRESMAX(**kwargs)
    return model

@register_model
def contrastive_resmaxv1(pretrained=False, **kwargs):
    """
    Registry function to create a ContrastiveRESMAX model
    via timm's create_model API.
    """
    for key in ["pretrained_cfg", "pretrained_cfg_overlay", "drop_rate"]:
        kwargs.pop(key, None)

    for key, val in kwargs.items():
        print(key, val)

    if pretrained:
        pass
    
    model = ContrastiveRESMAXV1(**kwargs)
    return model
@register_model
def resmax_bypass(pretrained=False, **kwargs):
    #deleting some kwargs that are messing up training
    try:
        del kwargs["pretrained_cfg"]
        del kwargs["pretrained_cfg_overlay"]
        del kwargs["drop_rate"]
    except:
        pass
    model = RESMAX_Bypass(**kwargs)
    if pretrained:
        pass
    return model

@register_model
def chresmax_v3_s(pretrained=False, **kwargs):
    """
    Registry function to create a CHRESMAX_V3_S model
    via timm's create_model API.
    """
    for key in ["pretrained_cfg", "pretrained_cfg_overlay", "drop_rate"]:
        kwargs.pop(key, None)

    for key, val in kwargs.items():
        print(key, val)

    model = CHRESMAX_V3_S(**kwargs) 
    return model
