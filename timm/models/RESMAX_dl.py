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
from .RESMAX import S1_Res, S2_Res, S2b_Res, S3_Res, pad_to_size





class RESMAX_V2_2_DL(nn.Module):
    def __init__(self, num_classes=1000, big_size=322, small_size=227, in_chans=3, 
                 ip_scale_bands=1, classifier_input_size=13312, contrastive_loss=False, pyramid=False,
                 bypass=False, main_route=False,
                 c_scoring='v2',
                 **kwargs):
        """
        smartly choose band in bypass use c score
        """
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
        super(RESMAX_V2_2_DL, self).__init__()

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
                nn.Conv2d(1024, 256, kernel_size=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((6, 6))
            )
            self.c2b_score = C_scoring2_optimized_debug(
                num_channels=1024,
                pool_func1=nn.MaxPool2d(kernel_size=3, stride=2),
                pool_func2=nn.MaxPool2d(kernel_size=4, stride=3),
                global_scale_pool=True
            )

        self.s3 = S3_Res()
        if self.ip_scale_bands > 6: # It was 4 before, this would only affect inference, during training was always below 4.
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
    def make_reference_scale(self, x, num_scale_bands, scale_band):
        base_image_size = int(x.shape[-1])
        scale_factor = 4  # exponent factor for scaling
        image_scales = get_ip_scales(num_scale_bands, base_image_size, scale_factor)
        x_reference = torch.zeros_like(x)
        for i in range(len(scale_band)):
            x_i = x[i:i+1]  # Keep batch dimension
            resize_size = int(image_scales[scale_band[i]])
            x_i = F.interpolate(x_i, size=(resize_size, resize_size), mode='bilinear', align_corners=False)
            # pad if it less than base_image_size or crop if it greater than base_image_size
            if x_i.shape[-1] < base_image_size:
                x_i = pad_to_size(x_i, (base_image_size, base_image_size))
            elif x_i.shape[-1] > base_image_size:
                center_crop = torchvision.transforms.CenterCrop(base_image_size)
                x_i = center_crop(x_i)
            x_reference[i] = x_i
           
        return x_reference

    def forward(self, x, main_route=False, pyramid=False,scale_band=None):

        if scale_band is not None:
            x_reference = self.make_reference_scale(x, self.ip_scale_bands, scale_band)
        else:
            x_reference = x

        if main_route:
            out = self.make_ip(x_reference, 3)
        else:
            out = self.make_ip(x_reference, self.ip_scale_bands)
        
        out = self.s1(out)
        out_c1 = self.c1(out)
        out = self.s2(out_c1)
        out_c2 = self.c2(out)
        
        if self.bypass:
            # import pdb; pdb.set_trace()
            bypass_value = self.s2b(out_c1)
            bypass_score = self.c2b_score(bypass_value)
            bypass_value = self.c2b_seq(bypass_score)
            bypass_value = bypass_value.reshape(bypass_value.size(0), -1)
        
        out = self.s3(out_c2)
        out = self.global_pool(out)
        if isinstance(out, list):
            out = out[-1]
        out = out.reshape(out.size(0), -1)

        if self.bypass:
            out = torch.cat([out, bypass_value], dim=1)
        
        out = self.fc(out)
        out = self.fc1(out)
        out = self.fc2(out)

        if self.contrastive_loss:
            if self.bypass:
                return out, out_c1, out_c2, bypass_value
            else:
                return out, out_c1, out_c2

        return out

"""
choose smartly in bypass, using supervision to choose reference scale. 
"""
class CHRESMAX_V3_2_DL(nn.Module):
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
        self.model_backbone = RESMAX_V2_2_DL(
            num_classes=num_classes,
            in_chans=in_chans,
            ip_scale_bands=self.ip_scale_bands,
            classifier_input_size=classifier_input_size,
            contrastive_loss=self.contrastive_loss,
            bypass=bypass,
        )

    def forward(self, x, scale_band=None,testing=False):
        """
        Creates two streams (original + random-scaled) for scale-consistency training.
        Returns:
            (output_of_stream1, correct_scale_loss)
        """
        # stream 1 (original scale)
        
        result = self.model_backbone(x, main_route=True, scale_band=scale_band)
        if self.bypass:
            stream_1_output, stream_1_c1_feats, stream_1_c2_feats, stream_1_bypass = result
        else:
            stream_1_output, stream_1_c1_feats, stream_1_c2_feats = result
        if testing:
            return stream_1_output, 0 
 
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

            center_crop = torchvision.transforms.CenterCrop(img_hw)
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

        correct_scale_loss = (c1_correct_scale_loss + c2_correct_scale_loss + bypass_correct_scale_loss)/3 + 0.1 * out_correct_scale_loss  

        return stream_1_output, correct_scale_loss


@register_model
def resmax_v2_sl(pretrained=False, **kwargs):
    
    try:
        del kwargs["pretrained_cfg"]
        del kwargs["pretrained_cfg_overlay"]
        del kwargs["drop_rate"]
    except:
        pass
    model = RESMAX_V2_2_DL(**kwargs)
    if pretrained:
        pass
    return model


@register_model
def chresmax_v3_2_dl(pretrained=False, **kwargs):
    try:
        del kwargs["pretrained_cfg"]
        del kwargs["pretrained_cfg_overlay"]
        del kwargs["drop_rate"]
    except:
        pass
    model = CHRESMAX_V3_2_DL(**kwargs)
    if pretrained:
        pass
    return model

