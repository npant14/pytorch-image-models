import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import multiprocessing

# Only import what's needed
from ._registry import register_model
from .HMAX import get_ip_scales
from .ALEXMAX import S1, S2, S3

# Only set start method if it hasn't been set
if not multiprocessing.get_start_method(allow_none=True):
    try:
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass

class CScaleSelectFC(nn.Module):
    """
    A 'C' layer that does scale selection via an FC for AlexMax.
    
    Steps:
      1) For each scale, do a conv to unify channels => out_chans.
      2) Resize all features to a common size using learnable resizing layers
      3) Global pool => shape (N, out_chans).
      4) Concatenate => FC => scale logits => softmax => alpha_i.
      5) Weighted sum of scale feature maps => fused.
      6) Return [fused], scale_logits.
    """
    def __init__(self, in_chans=96, out_chans=96, num_scales=2, conv_ks=3, resize_kernel_1=1, resize_kernel_2=3):
        super().__init__()
        self.num_scales = num_scales
        self.out_chans = out_chans
        
        # A conv for each scale
        self.convs = nn.Conv2d(in_chans, out_chans, kernel_size=conv_ks, padding=1)
        self.bns = nn.BatchNorm2d(out_chans)
        
        # Learnable resizing layers for each scale
        self.resizing_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(out_chans, out_chans, kernel_size=resize_kernel_1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_chans, out_chans, kernel_size=resize_kernel_2, padding=1)
            ) for _ in range(num_scales)
        ])
        
        # FC for scale selection - Changed to outpu num_scales instead of hardcoded 10
        self.fc = nn.Linear(out_chans * num_scales, num_scales)

    def forward(self, x_list):
        """
        x_list: list[Tensor] of length num_scales
            each => shape [N, in_chans, H, W]
        Returns:
            ([fused_feature], scale_logits)
              fused_feature => shape [N, out_chans, H, W] 
        """
        # conv each scale
        feats = []
        for i in range(self.num_scales):
            x = F.relu(self.bns(self.convs(x_list[i])))
            feats.append(x)
        
        # Resize all features to the middle scale size
        middle_idx = len(feats) // 2
        target_size = feats[middle_idx].shape[-2:]
        
        # Resize and apply learnable resizing layers
        resized_feats = []
        for i, feat in enumerate(feats):
            feat_resized = F.interpolate(feat, size=target_size, mode='bilinear', align_corners=False)
            feat_resized = self.resizing_layers[i](feat_resized)
            resized_feats.append(feat_resized)
        del feats
        # global avg pool => shape [N, out_chans]
        pooled_list = [f.mean(dim=(2,3)) for f in resized_feats]
        
        # concat => shape [N, out_chans * num_scales]
        concat_pooled = torch.cat(pooled_list, dim=1)
        
        # FC => scale logits => [N, num_scales]
        scale_logits = self.fc(concat_pooled)
        
        # softmax => scale weights
        alpha = F.softmax(scale_logits, dim=1)  # [N, num_scales]
        
        # Weighted sum of feats => fused
        N, C, H, W = resized_feats[0].shape
        alpha_5d = alpha.view(N, self.num_scales, 1, 1, 1)  # => [N, S, 1, 1, 1]
        stack_feats = torch.stack(resized_feats, dim=1)     # => [N, S, C, H, W]
        fused_feat = (alpha_5d * stack_feats).sum(dim=1)    # => [N, C, H, W]
        
        return [fused_feat], scale_logits



class BypassPath(nn.Module):
    """
    Minimal bypass from the C1 output (fused single scale).
    We'll match the spatial dimensions of S3's output using adaptive pooling
    or interpolation to ensure proper merging.
    """
    def __init__(self, in_chans=96, out_chans=256):
        super().__init__()
        # e.g. reduce from 96->256 with some stride 2 convs, etc.
        self.conv1 = nn.Conv2d(in_chans, 128, kernel_size=3, stride=2, padding=1)
        self.bn1   = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128, out_chans, kernel_size=3, stride=1, padding=1)  # Changed stride to 1
        self.bn2   = nn.BatchNorm2d(out_chans)

    def forward(self, x):
        # x => [N,96,Hc1,Wc1]
        out = F.relu(self.bn1(self.conv1(x)))  # First downsampling
        out = F.relu(self.bn2(self.conv2(out))) # No downsampling here
        return out


class AlexMaxBypassFC(nn.Module):
    """
    Example: S1->C1(fc scale selection)->S2->C2->S3->(global pool), with a bypass skipping S2/C2,
    merging with S3's output at the end.
    """
    def __init__(self, 
                 in_chans=3,
                 num_classes=1000,
                 contrastive_loss=False,
                 ip_scale_bands=2,
                 **kwargs):
        super().__init__()
        
        self.contrastive_loss = contrastive_loss
        self.ip_scale_bands = ip_scale_bands

        # S1
        self.s1= S1(kernel_size=11, stride=4, padding=0)
        # C1 with scale selection
        self.c1 = CScaleSelectFC(in_chans=96, out_chans=96, num_scales=ip_scale_bands)
        
        # S2
        self.s2 = S2(kernel_size=3, stride=1, padding=2)
        # C2 (purely spatial if you want - example: a maxpool for downsample)
        self.c2_pool = nn.MaxPool2d(kernel_size=3, stride=2)
        
        # S3
        self.s3 = S3()

        # Bypass path from c1 output => merges after s3
        self.bypass = BypassPath(in_chans=96, out_chans=256)

        # final classifier
        # after merging main(256) + bypass(256) => 512 channels
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512*6*6, 4096),  # if final map is 6x6 -> 512*(6*6)
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes)
        )

    def make_ip(self, x, num_scale_bands, center=None):
        ## num_scale_bands = num images in IP - 1
        
        # For PIL Image, use size instead of shape
        if isinstance(x, Image.Image):
            base_image_size = x.size[-1]
        else:
            base_image_size = int(x.shape[-1])
        scale = 4   ## factor in exponenet

        image_scales = get_ip_scales(num_scale_bands, base_image_size, scale)
        
        if len(image_scales) > 1:
            image_pyramid = []
            for i_s in image_scales:
                i_s = int(i_s)
                
                if center is not None:
                    # center shape: [batch_size, 2] where 2 is (x, y)
                    batch_size = center.shape[0]
                    
                    # Calculate crop window size for each image in batch
                    if isinstance(x, Image.Image):
                        width, height = x.size
                    else:
                        height, width = x.shape[2], x.shape[3]
                    crop_size = min(height, width)
                    
                    # Create a list to store individual cropped images
                    cropped_images = []
                    
                    # Process each image in the batch
                    for b in range(batch_size):
                        # Convert tensor values to Python scalars
                        center_x = center[b, 0].item()
                        center_y = center[b, 1].item()
                        
                        # Calculate crop boundaries for this image
                        start_x = max(0, int(center_x - crop_size // 2))
                        start_y = max(0, int(center_y - crop_size // 2))
                        end_x = min(width, start_x + crop_size)
                        end_y = min(height, start_y + crop_size)
                        
                        # Adjust start positions if we hit the boundaries
                        if end_x - start_x < crop_size:
                            start_x = max(0, end_x - crop_size)
                        if end_y - start_y < crop_size:
                            start_y = max(0, end_y - crop_size)
                        
                        # Crop the image
                        if isinstance(x, Image.Image):
                            # Convert PIL Image to tensor for this crop
                            img_tensor = transforms.ToTensor()(x)
                            img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension
                            cropped = img_tensor[:, :, start_y:end_y, start_x:end_x]
                        else:
                            cropped = x[b:b+1, :, start_y:end_y, start_x:end_x]
                        
                        cropped_images.append(cropped)
                    
                    # Stack all cropped images back into a batch
                    cropped_batch = torch.cat(cropped_images, dim=0)
                    
                    # Resize the entire batch at once
                    interpolated_img = F.interpolate(cropped_batch, size=(i_s, i_s), mode='bilinear')
                else:
                    if isinstance(x, Image.Image):
                        # Convert PIL Image to tensor for interpolation
                        x_tensor = transforms.ToTensor()(x)
                        x_tensor = x_tensor.unsqueeze(0)  # Add batch dimension
                        interpolated_img = F.interpolate(x_tensor, size=(i_s, i_s), mode='bilinear')
                    else:
                        interpolated_img = F.interpolate(x, size=(i_s, i_s), mode='bilinear')

                image_pyramid.append(interpolated_img)
            return image_pyramid
        else: 
            if isinstance(x, Image.Image):
                # Convert single PIL Image to tensor
                x_tensor = transforms.ToTensor()(x)
                x_tensor = x_tensor.unsqueeze(0)  # Add batch dimension
                return [x_tensor]
            return [x]

    def forward(self, x, main_route=False, center=None):
        
        # 1) Multi-scale input
        if main_route:
            out = self.make_ip(x, 2, center)
        else:
            out = self.make_ip(x, self.ip_scale_bands, center)
       

        # 2) S1
        out = self.s1(out)  # list of length num_scales => each [N,96,Hs1,Ws1]
       

        # 3) C1 with scale selection => returns [fused], scale_logits
        c1_out, scale_logits = self.c1(out)
        del out 
        # c1_out => list with 1 element => shape [N,96,Hc1,Wc1]
        fused_c1 = c1_out[0]
        
        # Bypass path
        bypass_out = self.bypass(fused_c1)  # => [N,256, Hc1/4, Wc1/4]

        # 4) S2 => returns [N,256,Hs2,Ws2]
        s2_out = self.s2(c1_out)  # => list with 1 element [N,256,Hs2,Ws2]
        fused_s2 = s2_out[0]
        
        # 5) C2 => purely spatial pool or a normal C. We'll do just a pool example
        c2_out = self.c2_pool(fused_s2)  # => [N,256,Hs2/2,Ws2/2]
        # wrap it in a list for S3
        c2_out_list = [c2_out]

        # 6) S3
        s3_out = self.s3(c2_out_list)  # => [ [N,256,Hs3,Ws3] ]
        fused_s3 = s3_out[0]

        # 7) Merge with bypass => cat
        # Make sure bypass_out shape matches fused_s3
        # If S2 + C2 each did stride=2 => total stride=4 from c1 => you want the bypass to do the same
        bypass_out = F.interpolate(bypass_out, size=fused_s3.shape[-2:], mode='bilinear', align_corners=False)
        merged = torch.cat([fused_s3, bypass_out], dim=1)  # => [N,512,Hfinal,Wfinal]

        # 8) global pool => assume final is 6x6
        # or if not 6x6, adapt your fc accordingly
        # Just do adaptive pool to 6x6
        out_pool = F.adaptive_avg_pool2d(merged, (6,6))  # => [N,512,6,6]
        out_flat = out_pool.view(out_pool.size(0), -1)   # => [N,512*6*6]

        # 9) final fc
        out = self.fc(out_flat)  # => [N,num_classes]

        if self.contrastive_loss:
            return out, scale_logits, fused_s3, fused_c1 
        else:
            return out, scale_logits

class ChAlexMaxBypassFC(nn.Module):
    """
    Example: S1->C1(fc scale selection)->S2->C2->S3->(global pool), with a bypass skipping S2/C2,
    merging with S3's output at the end.
    """
    def __init__(self, 
                 in_chans=3,
                 num_classes=1000,
                 contrastive_loss=False,
                 ip_scale_bands=2,
                 **kwargs):
        super().__init__()
        
        self.contrastive_loss = contrastive_loss
        self.ip_scale_bands = ip_scale_bands

        # S1
        self.s1= S1(kernel_size=11, stride=4, padding=0)
        # C1 with scale selection
        self.c1 = CScaleSelectFC(in_chans=96, out_chans=96, num_scales=ip_scale_bands)
        
        # S2
        self.s2 = S2(kernel_size=3, stride=1, padding=2)
        # C2 (purely spatial if you want - example: a maxpool for downsample)
        self.c2_pool = nn.MaxPool2d(kernel_size=3, stride=2)
        
        # S3
        self.s3 = S3()

        # Bypass path from c1 output => merges after s3
        self.bypass = BypassPath(in_chans=96, out_chans=256)

        # final classifier
        # after merging main(256) + bypass(256) => 512 channels
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512*6*6, 4096),  # if final map is 6x6 -> 512*(6*6)
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes)
        )

    def make_ip(self, x, num_scale_bands, center=None):
        ## num_scale_bands = num images in IP - 1
        
        # For PIL Image, use size instead of shape
        if isinstance(x, Image.Image):
            base_image_size = x.size[-1]
        else:
            base_image_size = int(x.shape[-1])
        scale = 4   ## factor in exponenet

        image_scales = get_ip_scales(num_scale_bands, base_image_size, scale)
        
        if len(image_scales) > 1:
            image_pyramid = []
            for i_s in image_scales:
                i_s = int(i_s)
                
                if center is not None:
                    # center shape: [batch_size, 2] where 2 is (x, y)
                    batch_size = center.shape[0]
                    
                    # Calculate crop window size for each image in batch
                    if isinstance(x, Image.Image):
                        width, height = x.size
                    else:
                        height, width = x.shape[2], x.shape[3]
                    crop_size = min(height, width)
                    
                    # Create a list to store individual cropped images
                    cropped_images = []
                    
                    # Process each image in the batch
                    for b in range(batch_size):
                        # Convert tensor values to Python scalars
                        center_x = center[b, 0].item()
                        center_y = center[b, 1].item()
                        
                        # Calculate crop boundaries for this image
                        start_x = max(0, int(center_x - crop_size // 2))
                        start_y = max(0, int(center_y - crop_size // 2))
                        end_x = min(width, start_x + crop_size)
                        end_y = min(height, start_y + crop_size)
                        
                        # Adjust start positions if we hit the boundaries
                        if end_x - start_x < crop_size:
                            start_x = max(0, end_x - crop_size)
                        if end_y - start_y < crop_size:
                            start_y = max(0, end_y - crop_size)
                        
                        # Crop the image
                        if isinstance(x, Image.Image):
                            # Convert PIL Image to tensor for this crop
                            img_tensor = transforms.ToTensor()(x)
                            img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension
                            cropped = img_tensor[:, :, start_y:end_y, start_x:end_x]
                        else:
                            cropped = x[b:b+1, :, start_y:end_y, start_x:end_x]
                        
                        cropped_images.append(cropped)
                    
                    # Stack all cropped images back into a batch
                    cropped_batch = torch.cat(cropped_images, dim=0)
                    
                    # Resize the entire batch at once
                    interpolated_img = F.interpolate(cropped_batch, size=(i_s, i_s), mode='bilinear')
                else:
                    if isinstance(x, Image.Image):
                        # Convert PIL Image to tensor for interpolation
                        x_tensor = transforms.ToTensor()(x)
                        x_tensor = x_tensor.unsqueeze(0)  # Add batch dimension
                        interpolated_img = F.interpolate(x_tensor, size=(i_s, i_s), mode='bilinear')
                    else:
                        interpolated_img = F.interpolate(x, size=(i_s, i_s), mode='bilinear')

                image_pyramid.append(interpolated_img)
            return image_pyramid
        else: 
            if isinstance(x, Image.Image):
                # Convert single PIL Image to tensor
                x_tensor = transforms.ToTensor()(x)
                x_tensor = x_tensor.unsqueeze(0)  # Add batch dimension
                return [x_tensor]
            return [x]

    def forward(self, x, main_route=False, center=None):
        
        # 1) Multi-scale input
        if main_route:
            out = self.make_ip(x, 2, center)
        else:
            out = self.make_ip(x, self.ip_scale_bands, center)
       

        # 2) S1
        out = self.s1(out)  # list of length num_scales => each [N,96,Hs1,Ws1]
       

        # 3) C1 with scale selection => returns [fused], scale_logits
        c1_out, scale_logits = self.c1(out)
        del out 
        # c1_out => list with 1 element => shape [N,96,Hc1,Wc1]
        fused_c1 = c1_out[0]
        
        # Bypass path
        bypass_out = self.bypass(fused_c1)  # => [N,256, Hc1/4, Wc1/4]

        # 4) S2 => returns [N,256,Hs2,Ws2]
        s2_out = self.s2(c1_out)  # => list with 1 element [N,256,Hs2,Ws2]
        fused_s2 = s2_out[0]
        
        # 5) C2 => purely spatial pool or a normal C. We'll do just a pool example
        c2_out = self.c2_pool(fused_s2)  # => [N,256,Hs2/2,Ws2/2]
        # wrap it in a list for S3
        c2_out_list = [c2_out]

        # 6) S3
        s3_out = self.s3(c2_out_list)  # => [ [N,256,Hs3,Ws3] ]
        fused_s3 = s3_out[0]

        # 7) Merge with bypass => cat
        # Make sure bypass_out shape matches fused_s3
        # If S2 + C2 each did stride=2 => total stride=4 from c1 => you want the bypass to do the same
        bypass_out = F.interpolate(bypass_out, size=fused_s3.shape[-2:], mode='bilinear', align_corners=False)
        merged = torch.cat([fused_s3, bypass_out], dim=1)  # => [N,512,Hfinal,Wfinal]

        # 8) global pool => assume final is 6x6
        # or if not 6x6, adapt your fc accordingly
        # Just do adaptive pool to 6x6
        out_pool = F.adaptive_avg_pool2d(merged, (6,6))  # => [N,512,6,6]
        out_flat = out_pool.view(out_pool.size(0), -1)   # => [N,512*6*6]

        # 9) final fc
        out = self.fc(out_flat)  # => [N,num_classes]

        if self.contrastive_loss:
            return out, scale_logits, fused_s3, fused_c1 
        else:
            return out, scale_logits
# Memory-efficient version
class CScaleSelectFCClean(nn.Module):
    """
    Memory-efficient version of CScaleSelectFC
    """
    def __init__(self, in_chans=96, out_chans=96, num_scales=2, conv_ks=3, resize_kernel_1=1, resize_kernel_2=3):
        super().__init__()
        self.num_scales = num_scales
        self.out_chans = out_chans
        
        # Single conv and bn for all scales to save memory
        self.conv = nn.Conv2d(in_chans, out_chans, kernel_size=conv_ks, padding=1)
        self.bn = nn.BatchNorm2d(out_chans)
        
        # Single resizing layer for all scales
        self.resizing_layer = nn.Sequential(
            nn.Conv2d(out_chans, out_chans, kernel_size=resize_kernel_1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_chans, out_chans, kernel_size=resize_kernel_2, padding=1)
        )
        
        # FC layer - fixed to match input dimensions (out_chans * num_scales)
        self.fc = nn.Linear(out_chans * (num_scales+1), (num_scales+1))

    def forward(self, x_list):
        # Process each scale one at a time to save memory
        middle_idx = len(x_list) // 2
        target_size = None
        pooled_features = []
        all_feats = []
        
        # First pass to get target size
        with torch.no_grad():
            middle_feat = self.conv(x_list[middle_idx])
            target_size = middle_feat.shape[-2:]
            del middle_feat
        
        # Process each scale
        
        for i, x in enumerate(x_list):
            # Conv + BN
            feat = F.relu(self.bn(self.conv(x)))
            
            # Resize to target size
            if feat.shape[-2:] != target_size:
                feat = F.interpolate(feat, size=target_size, mode='bilinear', align_corners=False)
            
            # Apply resizing layer
            feat = self.resizing_layer(feat)
            
            # Global pool
            pooled = feat.mean(dim=(2,3))  # Shape: [N, out_chans]
            pooled_features.append(pooled)
            
            # Store feature for later stacking
            all_feats.append(feat)
            
            del feat
        
        # Stack all features at once
        stacked_feats = torch.stack(all_feats, dim=1)  # [N, num_scales, C, H, W]
        del all_feats
        
        # Concatenate pooled features - ensure correct dimension
        concat_pooled = torch.cat(pooled_features, dim=1)  # Shape: [N, out_chans * num_scales]
        del pooled_features
        
        # Get scale logits and weights
        scale_logits = self.fc(concat_pooled)  # Shape: [N, num_scales]
        alpha = F.softmax(scale_logits, dim=1)
        
        
        # Reshape for fusion
        alpha_5d = alpha.view(alpha.size(0), alpha.size(1), 1, 1, 1)
       
        # Weighted sum
        fused_feat = (alpha_5d * stacked_feats).sum(dim=1)
        del stacked_feats, alpha_5d
        
        return [fused_feat], scale_logits

class AlexMaxBypassFCClean(nn.Module):
    """
    Memory-efficient version of AlexMaxBypassFC
    """
    def __init__(self, 
                 in_chans=3,
                 num_classes=1000,
                 contrastive_loss=False,
                 ip_scale_bands=2,
                 **kwargs):
        super().__init__()
        self.contrastive_loss = contrastive_loss
        self.ip_scale_bands = ip_scale_bands

        self.s1 = S1(kernel_size=11, stride=4, padding=0)
        self.c1 = CScaleSelectFCClean(in_chans=96, out_chans=96, num_scales=ip_scale_bands)
        self.s2 = S2(kernel_size=3, stride=1, padding=2)
        self.c2_pool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.s3 = S3()
        self.bypass = BypassPath(in_chans=96, out_chans=256)
        
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512*6*6, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes)
        )

    def make_ip(self, x, num_scale_bands, center=None):
        if isinstance(x, Image.Image):
            base_image_size = x.size[-1]
        else:
            base_image_size = int(x.shape[-1])
        scale = 4   ## factor in exponenet

        image_scales = get_ip_scales(num_scale_bands, base_image_size, scale)
        
        if len(image_scales) > 1:
            image_pyramid = []
            for i_s in image_scales:
                i_s = int(i_s)
                
                if center is not None:
                    # center shape: [batch_size, 2] where 2 is (x, y)
                    batch_size = center.shape[0]
                    
                    # Calculate crop window size for each image in batch
                    if isinstance(x, Image.Image):
                        width, height = x.size
                    else:
                        height, width = x.shape[2], x.shape[3]
                    crop_size = min(height, width)
                    
                    # Create a list to store individual cropped images
                    cropped_images = []
                    
                    # Process each image in the batch
                    for b in range(batch_size):
                        # Convert tensor values to Python scalars
                        center_x = center[b, 0].item()
                        center_y = center[b, 1].item()
                        
                        # Calculate crop boundaries for this image
                        start_x = max(0, int(center_x - crop_size // 2))
                        start_y = max(0, int(center_y - crop_size // 2))
                        end_x = min(width, start_x + crop_size)
                        end_y = min(height, start_y + crop_size)
                        
                        # Adjust start positions if we hit the boundaries
                        if end_x - start_x < crop_size:
                            start_x = max(0, end_x - crop_size)
                        if end_y - start_y < crop_size:
                            start_y = max(0, end_y - crop_size)
                        
                        # Crop the image
                        if isinstance(x, Image.Image):
                            # Convert PIL Image to tensor for this crop
                            img_tensor = transforms.ToTensor()(x)
                            img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension
                            cropped = img_tensor[:, :, start_y:end_y, start_x:end_x]
                        else:
                            cropped = x[b:b+1, :, start_y:end_y, start_x:end_x]
                        
                        cropped_images.append(cropped)
                    
                    # Stack all cropped images back into a batch
                    cropped_batch = torch.cat(cropped_images, dim=0)
                    
                    # Resize the entire batch at once
                    interpolated_img = F.interpolate(cropped_batch, size=(i_s, i_s), mode='bilinear')
                else:
                    if isinstance(x, Image.Image):
                        # Convert PIL Image to tensor for interpolation
                        x_tensor = transforms.ToTensor()(x)
                        x_tensor = x_tensor.unsqueeze(0)  # Add batch dimension
                        interpolated_img = F.interpolate(x_tensor, size=(i_s, i_s), mode='bilinear')
                    else:
                        interpolated_img = F.interpolate(x, size=(i_s, i_s), mode='bilinear')

                image_pyramid.append(interpolated_img)
            return image_pyramid
        else: 
            if isinstance(x, Image.Image):
                # Convert single PIL Image to tensor
                x_tensor = transforms.ToTensor()(x)
                x_tensor = x_tensor.unsqueeze(0)  # Add batch dimension
                return [x_tensor]
            return [x]
        

    def forward(self, x, main_route=False, center=None):
        # 1) Multi-scale input
        out = self.make_ip(x, 2 if main_route else self.ip_scale_bands, center)
        
        # 2) S1
        out = self.s1(out)
        del x  # Clean up input tensor
        
        # 3) C1 with scale selection
        c1_out, scale_logits = self.c1(out)
        del out
        fused_c1 = c1_out[0]
        
        # Store c1 features for contrastive loss if needed
        c1_features = fused_c1.clone() if self.contrastive_loss else None
        
        # Bypass path
        bypass_out = self.bypass(fused_c1)
        
        # 4) S2
        s2_out = self.s2(c1_out)
        del c1_out
        fused_s2 = s2_out[0]
        del s2_out
        
        # 5) C2
        c2_out = self.c2_pool(fused_s2)
        del fused_s2
        
        # 6) S3
        s3_out = self.s3([c2_out])
        del c2_out
        fused_s3 = s3_out[0]
        del s3_out
        
        # Store s3 features for contrastive loss if needed
        s3_features = fused_s3.clone() if self.contrastive_loss else None
        
        # 7) Merge with bypass
        bypass_out = F.interpolate(bypass_out, size=fused_s3.shape[-2:], 
                                 mode='bilinear', align_corners=False)
        merged = torch.cat([fused_s3, bypass_out], dim=1)
        del fused_s3, bypass_out
        
        # 8) Global pool
        out_pool = F.adaptive_avg_pool2d(merged, (6,6))
        del merged
        out_flat = out_pool.view(out_pool.size(0), -1)
        del out_pool
        
        # 9) Final fc
        out = self.fc(out_flat)
        del out_flat
        
        if self.contrastive_loss:
            return out, scale_logits, s3_features, c1_features
        return out, scale_logits

@register_model
def alexmax_bypass_dl_clean(pretrained=False, **kwargs):
    try:
        del kwargs["pretrained_cfg"]
        del kwargs["pretrained_cfg_overlay"]
        del kwargs["drop_rate"]
    except:
        pass
    if pretrained:
        raise ValueError("No pretrained model available for ALEXMAX_BYPASS_CLEAN")
    model = AlexMaxBypassFCClean(**kwargs)
    return model

@register_model
def chalexmax_bypass_dl(pretrained=False, **kwargs):
    try:
        del kwargs["pretrained_cfg"]
        del kwargs["pretrained_cfg_overlay"]
        del kwargs["drop_rate"]
    except:
        pass
    if pretrained:
        raise ValueError("No pretrained model available for CHALEXMAX_BYPASS")
    model = ChAlexMaxBypassFC(**kwargs)
    return model

@register_model
def alexmax_bypass_dl(pretrained=False, **kwargs):
    try:
        del kwargs["pretrained_cfg"]
        del kwargs["pretrained_cfg_overlay"]
        del kwargs["drop_rate"]
    except:
        pass
    if pretrained:
        raise ValueError("No pretrained model available for ALEXMAX_BYPASS")
    print(f"AlexMaxBypassFC created")
    model = AlexMaxBypassFC(**kwargs)
    return model

