"""
An implementation of HMAX:
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
from .HMAX import HMAX_from_Alexnet, HMAX_from_Alexnet_bypass,get_ip_scales,pad_to_size
from .ALEXMAX import S1, S2, S3, C, C_scoring, ConvScoring, soft_selection

class C_scoring2(nn.Module):
    # Spatial then Scale
    def __init__(self,
                 num_channels,
                 pool_func1=nn.MaxPool2d(kernel_size=3, stride=2),
                 pool_func2=nn.MaxPool2d(kernel_size=4, stride=3),
                 global_scale_pool=False):
        super(C_scoring2, self).__init__()
        self.pool1 = pool_func1
        self.pool2 = pool_func2
        self.global_scale_pool = global_scale_pool
        self.num_channels = num_channels
        self.scoring_conv = ConvScoring(num_channels)
        
        # Learnable resizing layers
        self.resizing_layers = nn.Sequential(
            nn.Conv2d(num_channels, num_channels, kernel_size=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        )

    def forward(self, x_pyramid):
        out = []
        if self.global_scale_pool:
            if len(x_pyramid) == 1:
                pooled = self.pool1(x_pyramid[0])
                _ = self.scoring_conv(pooled)  # Shape: [N]
                return pooled

            out = [self.pool1(x) for x in x_pyramid]
            final_size = out[len(out) // 2].shape[-2:]
            out_1 = F.interpolate(out[0], final_size, mode='bilinear', align_corners=False)
            out_1 = self.resizing_layers(out_1)
            for x in x_pyramid[1:]:
                temp = F.interpolate(x, final_size, mode='bilinear', align_corners=False)
                temp = self.resizing_layers(temp)
                score_out = self.scoring_conv(out_1)  # Shape: [N, H, W]
                score_temp = self.scoring_conv(temp)  # Shape: [N, H, W]
                scores = torch.stack([score_out, score_temp], dim=1).unsqueeze(2)  # Shape: [N, 2, 1, H, W]
                x = torch.stack([out_1, temp], dim=1)  # Shape: [N, 2, C, H, W]
                out_1 = soft_selection(scores, x)  # Differentiable selection
                del temp

        else:  # Not global pool
            if len(x_pyramid) == 1:
                pooled = self.pool1(x_pyramid[0])
                _ = self.scoring_conv(pooled)
                return [pooled]

            out_middle = self.pool2(x_pyramid[len(x_pyramid) // 2])
            final_size = out_middle.shape[-2:]

            for i in range(len(x_pyramid) - 1):
                x_1 = self.pool1(x_pyramid[i])
                x_2 = self.pool2(x_pyramid[i + 1])
                
                # Resize x_1 to final_size using learnable layers
                x_1 = F.interpolate(x_1, size=final_size, mode='bilinear', align_corners=False)
                x_1= self.resizing_layers(x_1)
                
                x_2 = F.interpolate(x_2, size=final_size, mode='bilinear', align_corners=False)
                x_2 = self.resizing_layers(x_2)
                
                # Compute scores using the learnable scoring function
                s_1 = self.scoring_conv(x_1)  # Shape: [N, 1, H, W]
                s_2 = self.scoring_conv(x_2)  # Shape: [N, 1, H, W]
                
                scores = torch.stack([s_1, s_2], dim=1)  # Shape: [N, 2, 1, H, W]

                x = torch.stack([x_1, x_2], dim=1)  # Shape: [N, 2, C, H, W]
                to_append = soft_selection(scores, x)
                out.append(to_append)

        return out
    
    
class ALEXMAX_v3(nn.Module):
    def __init__(self, num_classes=1000,base_size =322, in_chans=3, ip_scale_bands=1, classifier_input_size=13312, contrastive_loss=False,pyramid=False, **kwargs):
        self.num_classes = num_classes
        self.in_chans = in_chans
        self.contrastive_loss = contrastive_loss
        #ip_scale_bands: the number of scale BANDS (one less than the number of images in the pyramid)
        self.ip_scale_bands = ip_scale_bands
        self.pyramid = pyramid
        self.base_size = base_size
        super(ALEXMAX_v3, self).__init__()

        self.s1= S1(kernel_size=11, stride=4, padding=0)
      
        self.c1= C_scoring2(96,nn.MaxPool2d(kernel_size = 3, stride = 2), 
                               nn.MaxPool2d(kernel_size = 4, stride = 3), 
                               global_scale_pool=False)
        #self.s2b = S2b()
        #self.c2b = C(nn.MaxPool2d(kernel_size = 3, stride = 2), nn.MaxPool2d(kernel_size = 3, stride = 2), global_scale_pool=False)
        self.s2 = S2(kernel_size=3, stride=1, padding=2)
        self.c2 = C(nn.MaxPool2d(kernel_size = 3, stride = 2), nn.MaxPool2d(kernel_size = 3, stride = 2), global_scale_pool=False)
        self.s3 = S3()
        self.global_pool =  C(global_scale_pool=True)
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(classifier_input_size, 4096),
            nn.ReLU())
        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU())
        self.fc2= nn.Sequential(
            nn.Linear(4096, num_classes))
    def make_ip(self, x):
        ## num_scale_bands = num images in IP - 1
        num_scale_bands = self.ip_scale_bands
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
    def forward(self, x,pyramid=False):
        #resize image
        # always making pyramid to start with only two scales. 
        out = self.make_ip(x)
        
        ## should make SxBxCxHxW
        out_1 = self.s1(out)
        
        out_c1 = self.c1(out_1)
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
def alexmax_v3(pretrained=False, **kwargs):
    try:
        del kwargs["pretrained_cfg"]
        del kwargs["pretrained_cfg_overlay"]
        del kwargs["drop_rate"]
    except:
        pass
    if pretrained:
       raise ValueError("No pretrained model available for ALEXMAX_v3")
    model = ALEXMAX_v3(**kwargs)
    return model