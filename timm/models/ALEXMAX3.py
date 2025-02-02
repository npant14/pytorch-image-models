"""
An implementation of HMAX:
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
import scipy as sp
import time
import pdb
import random
from ._builder import build_model_with_cfg
from ._manipulate import checkpoint_seq
from ._registry import register_model, generate_default_cfgs
from .HMAX import HMAX_from_Alexnet, HMAX_from_Alexnet_bypass,get_ip_scales,pad_to_size
from .ALEXMAX import S1, S2, S3, C, C_scoring, ConvScoring, soft_selection

class C_scoring2(nn.Module):
    """
    C layer with learnable scoring function. 
    This layer is used in the ALEXMAX_v3 model.
    It resizes the image then applies convolutional filtering to the resized images.
    The scores are computed using a learnable scoring function.
    
    Args:
        num_channels (int): Number of input channels
        pool_func1 (nn.Module): Pooling function for the first input
        pool_func2 (nn.Module): Pooling function for the second input
        global_scale_pool (bool): Whether to use global scale pooling
        
    
    """
    
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
                pooled = self.pool2(x_pyramid[0])
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
    """
    ALEXMAX_v3 model. This model is an implementation of the HMAX model.
    It is based on the ALEXNET architecture.
    Implement only one S1 layer. The C1 layer is replaced with a learnable scoring function.
    Also it has resizing layers and a learnable scoring function.
    
    
    """
    
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
    
class CHALEXMAX3(nn.Module):
    def __init__(self, num_classes=1000,
                 base_size =322, 
                 in_chans=3, 
                 teacher_scale_bands=1,
                  student_scale_bands =3,
                 classifier_input_size=13312,
                **kwargs):
        super(CHALEXMAX3, self).__init__()
        self.num_classes = num_classes
        self.in_chans = in_chans
        self.contrastive_loss = True # this tells the trainer to expect a contrastive loss
        self.base_size = base_size
        self.student_scale_bands = student_scale_bands
        self.teacher_scale_bands = teacher_scale_bands
        self.classifier_input_size = classifier_input_size
        self.teacher = ALEXMAX_v3(base_size =base_size,
                                  ip_scale_bands = teacher_scale_bands,
                                  classifier_input_size=classifier_input_size, 
                                  contrastive_loss=True,pyramid=False)
        self.student = ALEXMAX_v3(base_size =base_size, 
                                  ip_scale_bands=student_scale_bands,
                                  classifier_input_size=classifier_input_size, 
                                  contrastive_loss=True,pyramid=False)
    
    def augment(self, x):
        scale_factor_list =   np.arange(-self.student_scale_bands//2 + 1, self.student_scale_bands//2 + 2) #[0.707, 0.841, 1, 1.189, 1.414]
        scale_factor_list = [2**(i/4) for i in scale_factor_list]
        
        scale_factor = random.choice(scale_factor_list)
        img_hw = x.shape[-1]
        new_hw = int(img_hw*scale_factor)
        x_rescaled = F.interpolate(x, size = (new_hw, new_hw), mode = 'bilinear').clamp(min=0, max=1)
        if new_hw <= img_hw:
            x_rescaled = pad_to_size(x_rescaled, (img_hw, img_hw))
        elif new_hw > img_hw:
            center_crop = torchvision.transforms.CenterCrop(img_hw)
            x_rescaled = center_crop(x_rescaled)
        return x_rescaled
    def correct_size(self, student_features,teacher_features):
        
        if student_features.shape[-1] > teacher_features.shape[-1]:
            center_crop = torchvision.transforms.CenterCrop(teacher_features.shape[-1])
            student_features = center_crop(student_features)
        elif student_features.shape[-1] < teacher_features.shape[-1]:
            student_features = pad_to_size(student_features, (teacher_features.shape[-1], teacher_features.shape[-1]))
        return student_features
    def forward(self, x):
        
        out_teacher, stream_teacher_c1, stream_teacher_c2 = self.teacher(x)
        x_rescaled = self.augment(x)
        out_student, stream_student_c1, stream_student_c2 = self.student(x_rescaled)
        out = out_teacher/2 + out_student / 2 
        stream_student_c1 = self.correct_size(stream_student_c1, stream_teacher_c1)
        stream_student_c2 = self.correct_size(stream_student_c2, stream_teacher_c2)
        correct_scale_loss = F.mse_loss(stream_teacher_c1, stream_student_c1) + F.mse_loss(stream_teacher_c2, stream_student_c2)
        return out, correct_scale_loss
    
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

@register_model
def chalexmax_v3(pretrained=False, **kwargs):
    try:
        del kwargs["pretrained_cfg"]
        del kwargs["pretrained_cfg_overlay"]
        del kwargs["drop_rate"]
    except:
        pass
    if pretrained:
       raise ValueError("No pretrained model available for CHALEXMAX3")
    model = CHALEXMAX3(**kwargs)
    return model