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


from ._builder import build_model_with_cfg
from ._manipulate import checkpoint_seq
from ._registry import register_model, generate_default_cfgs
from .ALEXMAX import C_scoring, C
from .ALEXMAX3 import C_scoring2
from .ALEXMAX3_optimized import C_scoring2_optimized
from .HMAX import get_ip_scales, pad_to_size


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
        if self.main_route:
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
                return out, out_c1[0], out_c2[0], bypass
            else:
                return out, out_c1[0], out_c2[0]

        return out


class CHRESMAX_V2(nn.Module):
    """
    Example student-teacher style model with scale-consistency loss,
    using RESMAX_V2 as the backbone.
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
            center_crop = torchvision.transforms.CenterCrop(img_hw)
            x_rescaled = center_crop(x_rescaled)

        # forward pass on the scaled input
        result = self.model_backbone(x_rescaled)
        if self.bypass:
            stream_2_output, stream_2_c1_feats, stream_2_c2_feats, stream_2_bypass = result
        else:
            stream_2_output, stream_2_c1_feats, stream_2_c2_feats = result

        # scale-consistency loss
        c1_correct_scale_loss = torch.mean(torch.abs(stream_1_c1_feats - stream_2_c1_feats))
        c2_correct_scale_loss = torch.mean(torch.abs(stream_1_c2_feats - stream_2_c2_feats))
        out_correct_scale_loss = torch.mean(torch.abs(stream_1_output - stream_2_output))

        if self.bypass:
            bypass_correct_scale_loss = torch.mean(torch.abs(stream_1_bypass - stream_2_bypass))
        else:
            bypass_correct_scale_loss = 0

        correct_scale_loss = c1_correct_scale_loss + c2_correct_scale_loss + 0.1 * out_correct_scale_loss + bypass_correct_scale_loss

        return stream_1_output, correct_scale_loss


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
        # for param in self.backbone.s3.parameters():
        #     param.requires_grad = True
        for param in self.backbone.c1.parameters():
            param.requires_grad = True
        for param in self.backbone.c2.parameters():
            param.requires_grad = True
        if bypass:
            for param in self.backbone.c2b_seq.parameters():
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
        c1_contrastive_loss = nt_xent_loss(c1_feats1, c1_feats2)
        c2_contrastive_loss = nt_xent_loss(c2_feats1, c2_feats2)
        
        # Compute output contrastive loss
        out_contrastive_loss = nt_xent_loss(out1, out2)
        
        # Compute bypass contrastive loss if needed
        if self.backbone.bypass:
            bypass_contrastive_loss = nt_xent_loss(bypass1, bypass2)
            total_loss = c1_contrastive_loss + c2_contrastive_loss + 0.1 * out_contrastive_loss + 0.1 * bypass_contrastive_loss
        else:
            total_loss = c1_contrastive_loss + c2_contrastive_loss + 0.1 * out_contrastive_loss
        
        return out1, total_loss
    

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
    if pretrained:
        pass
    return model


@register_model
def chresmax_v2(pretrained=False, **kwargs):
    """
    Registry function to create a CHALEXMAX_V3_3_optimized model
    via timm's create_model API.
    """
    for key in ["pretrained_cfg", "pretrained_cfg_overlay", "drop_rate"]:
        kwargs.pop(key, None)

    if pretrained:
        pass
    model = CHRESMAX_V2(**kwargs)
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