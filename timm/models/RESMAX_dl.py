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
from ._builder import build_model_with_cfg
from ._manipulate import checkpoint_seq
from ._registry import register_model, generate_default_cfgs
from .ALEXMAX import C_scoring, C
from .ALEXMAX3 import C_scoring2
from .ALEXMAX3_optimized import C_scoring2_optimized, C_scoring2_optimized_debug
from .HMAX import get_ip_scales
from .ALEXMAX3_dl import CScaleSelectFC, CScaleSelectFC_v2, CScaleSelectFC_v3
from .RESMAX import S1_Res, S2_Res, S2b_Res, S3_Res

class RESMAX_V2_SL(nn.Module):
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
        super(RESMAX_V2_SL, self).__init__()

        self.s1 = S1_Res()
        
        self.c1 = C_scoring2_optimized(
            num_channels=96,
            pool_func1=nn.MaxPool2d(kernel_size=3, stride=2),
            pool_func2=nn.MaxPool2d(kernel_size=4, stride=3),
            skip=1,
            global_scale_pool=False
        )
        self.s2 = S2_Res()
        self.c2 = C_scoring2_optimized(
            num_channels=256,
            pool_func1=nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            pool_func2=nn.MaxPool2d(kernel_size=6, stride=2),
            resize_kernel_1=3,
            resize_kernel_2=1,
            skip=1,
            global_scale_pool=False
        )
        if self.bypass:
            self.s2b = S2b_Res()
            self.c2b = nn.Sequential(
                nn.MaxPool2d(kernel_size=3, stride=2),
                nn.MaxPool2d(kernel_size=3, stride=2),
                nn.Conv2d(1024, 256, kernel_size=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True)
            )
        
        self.s3 = S3_Res()
        self.c3 = CScaleSelectFC_v3(
            in_chans=256,
            out_chans=256,
            last_features=1024,
            num_scales=4,
            conv_ks=3,
            resize_kernel_1=1,
            resize_kernel_2=3
            )
        self.c4 = C(global_scale_pool=True)
        self.pool = nn.AdaptiveAvgPool2d((6,6))
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
        
        out = self.make_ip(x, self.ip_scale_bands)
        
        out = self.s1(out)
        out_c1 = self.c1(out)
        out = self.s2(out_c1)
        out_c2 = self.c2(out)
        if self.bypass:
            bypass = self.s2b(out_c1)
            bypass = self.c2b(bypass[-1])
            
            bypass = bypass.reshape(bypass.size(0), -1)
        
        out = self.s3(out_c2)
        
        out,scale_logits = self.c3(out)

        out = self.c4(out)
        
        
        
        if isinstance(out, list):
            out = out[0]
        out = self.pool(out)
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

        return out,scale_logits

@register_model
def resmax_v2_sl(pretrained=False, **kwargs):
    
    try:
        del kwargs["pretrained_cfg"]
        del kwargs["pretrained_cfg_overlay"]
        del kwargs["drop_rate"]
    except:
        pass
    model = RESMAX_V2_SL(**kwargs)
    if pretrained:
        pass
    return model



