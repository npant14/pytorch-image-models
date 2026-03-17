import random
import torch
import numpy as np
from typing import Iterator
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from typing import List, Dict


############################## Pasupathy Code ##############################




############################### Korean ##############################

class FeatureExtractor(nn.Module):
    """Register forward hooks on specified layers and return their outputs.

    Usage:
        fe = FeatureExtractor(model, ["layer.name"])  # names from model.named_modules()
        feats = fe(x)  # dict[layer_name] -> tensor output
    """
    def __init__(self, model: nn.Module, layers: List[str]):
        super().__init__()
        self.model = model
        self.layers = layers
        self._features: Dict[str, torch.Tensor] = {layer: torch.empty(0) for layer in layers}

        named = dict([*self.model.named_modules()])
        for layer_id in layers:
            if layer_id not in named:
                raise KeyError(f"Layer '{layer_id}' not found in model.named_modules().")
            layer = named[layer_id]
            layer.register_forward_hook(self._save_outputs_hook(layer_id))

    def _save_outputs_hook(self, layer_id):
        def fn(_, __, output):
            self._features[layer_id] = output
        return fn

    def forward(self, x: torch.Tensor):
        _ = self.model(x)
        return self._features
    
class Invert:
    def __call__(self, sample):
        inverted_image = (-1 * sample) + 1
        return inverted_image
    
    
############################## Training Utils ##############################
    

def pad_batch_random(images, target_size):
    """
    Pad images with random positioning within the target size.
    
    Args:
        images (torch.Tensor): Input images of shape (batch_size, channels, height, width)
        target_size (tuple): Desired output size as (target_height, target_width)
    
    Returns:
        torch.Tensor: Padded images with random positioning
    """
    _, _, h, w = images.shape
    target_h, target_w = target_size

    # Calculate total padding needed
    pad_h = max(target_h - h, 0)
    pad_w = max(target_w - w, 0)

    # For each image in the batch, generate random padding
    batch_size = images.shape[0]
    padded_images = []
    
    for i in range(batch_size):
        # Randomly decide top/left padding
        pad_top = random.randint(0, pad_h)
        pad_left = random.randint(0, pad_w)
        
        # Bottom/right padding is whatever remains
        pad_bottom = pad_h - pad_top
        pad_right = pad_w - pad_left
        
        # Pad individual image
        padded_image = F.pad(
            images[i:i+1],
            (pad_left, pad_right, pad_top, pad_bottom),
            mode='constant',
        )
        padded_images.append(padded_image)
    
    # Stack all padded images back into a batch
    return torch.cat(padded_images, dim=0)


def pad_batch(images, target_size):
    _, _, h, w = images.shape
    target_h, target_w = target_size

    # Calculate padding
    pad_h = max(target_h - h, 0)
    pad_w = max(target_w - w, 0)

    # Calculate padding for each side
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    # Apply padding
    # Use reflection padding to avoid artifacts, or constant for default
    padded_images = F.pad(images, (pad_left, pad_right, pad_top, pad_bottom), mode='constant')

    return padded_images


def pad_to_size_gray(a, size, gray_val_float=0.5, gray_val_uint8=128):
    """
    Pads tensor `a` (B, C, H, W) to `size` with uniform gray background using F.pad.
    """
    current_size = a.shape[-2:]  # (H, W)
    pad_h = size[0] - current_size[0]
    pad_w = size[1] - current_size[1]

    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    if torch.is_floating_point(a):
        pad_val = gray_val_float
    else:
        pad_val = gray_val_uint8

    # Note: F.pad pads in (left, right, top, bottom) order
    a_padded = F.pad(a, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=pad_val)
    return a_padded


def pad_to_size_noise(a, size, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    Pads tensor `a` (B, C, H, W) to the given `size` (H_out, W_out) with Gaussian noise.
    Noise is generated per channel with the given mean and std (e.g., ImageNet stats).
    """
    current_size = a.shape[-2:]  # (H, W)
    pad_h = size[0] - current_size[0]
    pad_w = size[1] - current_size[1]

    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    dtype = a.dtype
    device = a.device
    B, C = a.shape[:2]

    # Create noise background
    canvas = torch.zeros((B, C, size[0], size[1]), dtype=dtype, device=device)
    for c in range(C):
        noise = torch.randn((B, 1, size[0], size[1]), dtype=dtype, device=device) * std[c] + mean[c]
        canvas[:, c:c+1, :, :] = noise

    # Paste input tensor in the center
    canvas[:, :, pad_top:pad_top + current_size[0], pad_left:pad_left + current_size[1]] = a
    return canvas


def pad_to_size_blue(a, size):
    """
    Pads tensor `a` (B, C, H, W) to the given `size` (H_out, W_out) with blue color.
    """
    current_size = a.shape[-2:]  # (H, W)
    pad_h = size[0] - current_size[0]
    pad_w = size[1] - current_size[1]

    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    # Create a blue canvas (R=0, G=0, B=1 if input is float; B=255 if uint8)
    dtype = a.dtype
    device = a.device
    B, C = a.shape[:2]
    blue_val = 1.0 if dtype == torch.float32 else 255
    canvas = torch.zeros((B, C, size[0], size[1]), dtype=dtype, device=device)
    if C == 3:
        canvas[:, 2, :, :] = blue_val  # Blue channel

    # Paste `a` in the center
    canvas[:, :, pad_top:pad_top + current_size[0], pad_left:pad_left + current_size[1]] = a
    return canvas


class RandomResizePad:
    def __init__(self, original_size=(227, 227), min_size=160, max_size=None):
        """
        Args:
            original_size (tuple): Original image size (height, width)
            min_size (int): Minimum size for random resize
        """
        self.original_size = original_size
        self.min_size = min_size
        if max_size is None:
            self.max_size = original_size[0]
            print("Max size not provided. Setting to original size.")
        else:
            self.max_size = max_size
        print("Random Resize range:", self.min_size, "-", self.max_size)

    def __call__(self, img):
        """
        Args:
            img (Tensor): Image tensor of shape (C, H, W)
        Returns:
            Tensor: Transformed image tensor of shape (C, H, W)
        """
        # Randomly choose new size
        new_size = random.randint(self.min_size, self.max_size)
        
        # Calculate aspect ratio
        orig_h, orig_w = self.original_size
        aspect_ratio = orig_w / orig_h
        
        # Determine new height and width maintaining aspect ratio
        if aspect_ratio > 1:
            new_w = new_size
            new_h = int(new_size / aspect_ratio)
        else:
            new_h = new_size
            new_w = int(new_size * aspect_ratio)
        
        # Resize image
        resized_img = TF.resize(img, (new_h, new_w))
        
        # Calculate padding
        pad_h = max(self.original_size[0] - new_h, 0)
        pad_w = max(self.original_size[1] - new_w, 0)
        
        pad_top = random.randint(0, pad_h)
        pad_left = random.randint(0, pad_w)
        
        # Bottom/right padding is whatever remains
        pad_bottom = pad_h - pad_top
        pad_right = pad_w - pad_left
        
        # Pad individual image
        padded_img = F.pad(
            resized_img,
            (pad_left, pad_right, pad_top, pad_bottom),
            mode='constant',
            value=0
        )

        return padded_img
    

class RandomCenterResizeCropPad:
    def __init__(self,
                 output_size=(227, 227),
                 scale_choices=[160, 192, 227, 270, 322, 382, 454],
                 mode='constant'):
        """
        Transform that handles different scale invariances.
        
        Args:
            output_size (tuple): The final output size (height, width) that the network expects
            scale (int): The scale invariance to test
                        If scale <= min(output_size), the image is resized to scale and center-padded
                        If scale > min(output_size), the image is resized to scale and center-cropped
        """
        self.output_size = output_size if isinstance(output_size, tuple) else (output_size, output_size)
        self.scale_choices = scale_choices
        self.mode = mode
        
    def __call__(self, img):
        """
        Args:
            img (Tensor): Image tensor of shape (C, H, W)
        Returns:
            Tensor: Transformed image tensor of shape (C, output_size[0], output_size[1])
        """
        scale = random.choice(self.scale_choices)

        # Get original image dimensions
        _, orig_h, orig_w = img.shape
        
        # Calculate aspect ratio
        aspect_ratio = orig_w / orig_h
        
        # Determine new height and width based on scale while maintaining aspect ratio
        # Set the smaller dimension to scale
        if aspect_ratio > 1:  # Width > Height (scale applies to height)
            new_h = scale
            new_w = int(scale * aspect_ratio)
        else:  # Height >= Width (scale applies to width)
            new_w = scale
            new_h = int(scale / aspect_ratio)
        
        # Resize image to the target scale
        resized_img = TF.resize(img, (new_h, new_w))
        
        # Case 1: If scale <= min(output_size), pad to output_size
        if scale <= min(self.output_size):
            # Calculate padding needed for each dimension
            pad_h = max(self.output_size[0] - new_h, 0)
            pad_w = max(self.output_size[1] - new_w, 0)
            
            # Calculate padding for each side (center padding)
            pad_top = pad_h // 2
            pad_bottom = pad_h - pad_top
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left
            
            if self.mode in ['replicate', 'circular', 'constant', 'reflect']:
                transformed_img = F.pad(
                    resized_img,
                    (pad_left, pad_right, pad_top, pad_bottom),
                    mode=self.mode
                )
            elif self.mode == 'gray':
                transformed_img = pad_to_size_gray(resized_img.unsqueeze(0), self.output_size).squeeze(0)
            elif self.mode == 'blue':
                transformed_img = pad_to_size_blue(resized_img.unsqueeze(0), self.output_size).squeeze(0)
            elif self.mode == 'noise':
                transformed_img = pad_to_size_noise(resized_img.unsqueeze(0), self.output_size).squeeze(0)
            else:
                raise ValueError(f"Unsupported padding mode: {self.mode}")
            
        # Case 2: If scale > min(output_size), center crop to output_size
        else:
            # Calculate crop coordinates
            crop_h = self.output_size[0]
            crop_w = self.output_size[1]
            
            # Calculate top-left coordinates for center crop
            top = (new_h - crop_h) // 2
            left = (new_w - crop_w) // 2
            
            # Apply center crop
            transformed_img = TF.crop(resized_img, top, left, crop_h, crop_w)
        
        return transformed_img
    

class CenterResizeCropPad:
    def __init__(self, output_size=(227, 227), scale=160, mode='constant'):
        """
        Transform that handles different scale invariances.
        
        Args:
            output_size (tuple): The final output size (height, width) that the network expects
            scale (int): The scale invariance to test
                        If scale <= min(output_size), the image is resized to scale and center-padded
                        If scale > min(output_size), the image is resized to scale and center-cropped
        """
        self.output_size = output_size if isinstance(output_size, tuple) else (output_size, output_size)
        self.scale = scale
        self.mode = mode
        
    def __call__(self, img):
        """
        Args:
            img (Tensor): Image tensor of shape (C, H, W)
        Returns:
            Tensor: Transformed image tensor of shape (C, output_size[0], output_size[1])
        """
        # Get original image dimensions
        _, orig_h, orig_w = img.shape
        
        # Calculate aspect ratio
        aspect_ratio = orig_w / orig_h
        
        # Determine new height and width based on scale while maintaining aspect ratio
        # Set the smaller dimension to scale
        if aspect_ratio > 1:  # Width > Height (scale applies to height)
            new_h = self.scale
            new_w = int(self.scale * aspect_ratio)
        else:  # Height >= Width (scale applies to width)
            new_w = self.scale
            new_h = int(self.scale / aspect_ratio)
        
        # Resize image to the target scale
        resized_img = TF.resize(img, (new_h, new_w))
        
        # Case 1: If scale <= min(output_size), pad to output_size
        if self.scale <= min(self.output_size):
            # Calculate padding needed for each dimension
            pad_h = max(self.output_size[0] - new_h, 0)
            pad_w = max(self.output_size[1] - new_w, 0)
            
            # Calculate padding for each side (center padding)
            pad_top = pad_h // 2
            pad_bottom = pad_h - pad_top
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left
            
            if self.mode in ['replicate', 'circular', 'constant', 'reflect']:
                transformed_img = F.pad(
                    resized_img,
                    (pad_left, pad_right, pad_top, pad_bottom),
                    mode=self.mode
                )
            elif self.mode == 'gray':
                transformed_img = pad_to_size_gray(resized_img.unsqueeze(0), self.output_size).squeeze(0)
            elif self.mode == 'blue':
                transformed_img = pad_to_size_blue(resized_img.unsqueeze(0), self.output_size).squeeze(0)
            elif self.mode == 'noise':
                transformed_img = pad_to_size_noise(resized_img.unsqueeze(0), self.output_size).squeeze(0)
            else:
                raise ValueError(f"Unsupported padding mode: {self.mode}")
            
        # Case 2: If scale > min(output_size), center crop to output_size
        else:
            # Calculate crop coordinates
            crop_h = self.output_size[0]
            crop_w = self.output_size[1]
            
            # Calculate top-left coordinates for center crop
            top = (new_h - crop_h) // 2
            left = (new_w - crop_w) // 2
            
            # Apply center crop
            transformed_img = TF.crop(resized_img, top, left, crop_h, crop_w)
        
        return transformed_img
    
class CenterCropPad:
    def __init__(self, output_size=(227, 227), crop_size=160, mode='constant'):
        """
        Transform that crops from center to crop_size, then pads back to output_size.
        This is different from CenterResizeCropPad which resizes first.
        
        Args:
            output_size (tuple): The final output size (height, width) 
            crop_size (int): The size to crop from center (receptive field size)
            mode (str): Padding mode ('constant', 'gray', 'blue', 'noise', etc.)
        """
        self.output_size = output_size if isinstance(output_size, tuple) else (output_size, output_size)
        self.crop_size = crop_size
        self.mode = mode
        
    def __call__(self, img):
        """
        Args:
            img (Tensor): Image tensor of shape (C, H, W)
        Returns:
            Tensor: Transformed image tensor of shape (C, output_size[0], output_size[1])
        """
        _, orig_h, orig_w = img.shape
        
        # Step 1: Center crop to crop_size
        # Calculate crop coordinates for center crop
        crop_h = min(self.crop_size, orig_h)
        crop_w = min(self.crop_size, orig_w)
        
        # Calculate top-left coordinates for center crop
        top = (orig_h - crop_h) // 2
        left = (orig_w - crop_w) // 2
        
        # Apply center crop
        cropped_img = TF.crop(img, top, left, crop_h, crop_w)
        
        # Step 2: Pad the cropped image back to output_size
        # Calculate padding needed for each dimension
        pad_h = max(self.output_size[0] - crop_h, 0)
        pad_w = max(self.output_size[1] - crop_w, 0)
        
        # Calculate padding for each side (center padding)
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        
        # Apply padding based on mode
        if self.mode in ['replicate', 'circular', 'constant', 'reflect']:
            transformed_img = F.pad(
                cropped_img,
                (pad_left, pad_right, pad_top, pad_bottom),
                mode=self.mode
            )
        elif self.mode == 'gray':
            transformed_img = pad_to_size_gray(cropped_img.unsqueeze(0), self.output_size).squeeze(0)
        elif self.mode == 'blue':
            transformed_img = pad_to_size_blue(cropped_img.unsqueeze(0), self.output_size).squeeze(0)
        elif self.mode == 'noise':
            transformed_img = pad_to_size_noise(cropped_img.unsqueeze(0), self.output_size).squeeze(0)
        else:
            raise ValueError(f"Unsupported padding mode: {self.mode}")
        
        return transformed_img

import os

class DataLoaderTransformWrapper:
    def __init__(self, dataloader: DataLoader, transform=None):
        """
        Args:
            dataloader (DataLoader): Original PyTorch DataLoader
            transform: Transform to apply to images
        """
        self.dataloader = dataloader
        self.transform = transform
        
        # Expose important DataLoader attributes
        self.sampler = dataloader.sampler
        
    def __len__(self):
        return self.dataloader.__len__()
    
    def __iter__(self) -> Iterator:
        """
        Iterates over the dataloader and applies the transform to the images.
        Assumes the first element of each batch is the images.
        """
        iterator = iter(self.dataloader)
        # samples_seen = 0

        for batch in iterator:
            # if samples_seen >= 10000:
            #     break
            if isinstance(batch, torch.Tensor):
                # If batch is just a tensor of images
                transformed_images = torch.stack([self.transform(img) for img in batch])
                # samples_seen += len(transformed_images)
                yield transformed_images
            elif isinstance(batch, (tuple, list)):
                # If batch is (images, labels) or similar
                images = batch[0]
                transformed_images = torch.stack([self.transform(img) for img in images])
                # samples_seen += len(transformed_images)
                yield (transformed_images,) + batch[1:]
                
    def set_transform(self, transform):
        """
        Update the transform
        """
        self.transform = transform

    # Forward any missing attributes to the underlying dataloader
    def __getattr__(self, name):
        return getattr(self.dataloader, name)

def visualize_transforms(img, scales, target_size=(322, 322), save_path="transform_modes_grid.png"):
    # Ensure single image
    if img.dim() == 4:
        img_tensor = img[0]
    else:
        img_tensor = img

    # Padding modes to visualize
    modes = ['noise', 'gray', 'blue', 'constant', 'replicate', 'reflect', 'circular']
    n_modes = len(modes)
    n_scales = len(scales)

    # Prepare subplot grid
    fig, axes = plt.subplots(n_modes, n_scales, figsize=(4 * n_scales, 4 * n_modes))

    for row_idx, mode in enumerate(modes):
        for col_idx, scale in enumerate(scales):
            transform = CenterResizeCropPad(output_size=target_size, scale=scale, mode=mode)
            transformed_tensor = transform(img_tensor)

            transformed_img = transformed_tensor.cpu().permute(1, 2, 0).numpy()
            transformed_img = np.clip(transformed_img, 0, 1)

            ax = axes[row_idx, col_idx] if n_modes > 1 else axes[col_idx]
            ax.imshow(transformed_img)
            ax.axis('off')

            if row_idx == 0:
                mode_type = "padded" if scale <= min(target_size) else "cropped"
                ax.set_title(f"Scale {scale} ({mode_type})")

            if col_idx == 0:
                ax.set_ylabel(mode, fontsize=12)

    plt.tight_layout()
    
    base_name = os.path.splitext(save_path)[0]
    path = os.path.join('visualize', base_name) # ./visualzie/transform_modes_grid/???
    os.makedirs(path, exist_ok=True)
    
    plt.savefig(os.path.join(path, save_path))
    print(f"Transform grid visualization saved to {os.path.join(path, save_path)}")
    
    
    for scale in scales:
        transform = CenterResizeCropPad(output_size=target_size, scale=scale, mode='constant')
        transformed = transform(img_tensor)

        # properly convert this time
        arr = transformed.cpu().permute(1, 2, 0).numpy()
        arr = np.clip(arr, 0, 1)
        
        plt.imsave(
            os.path.join(path, f"constant_{scale}.png"),
            arr,
            vmin=0, vmax=1        # make sure it knows your data is in [0,1]
        )


############################### Pasupathy Model Loaders ##############################

def load_chresmax_v3_2_abs(device='cuda'):
    """Load CHResMax v3.2 (absolute value) model"""
    from timm.models.RESMAX import chresmax_v3_2_abs
    
    checkpoint_path = '/oscar/data/tserre/xyu110/pytorch-output/train/0/final_versions/ip_3_chresmax_v3_2_abs_gpu_8_cl_0.1_ip_3_322_322_18432_c1[_6,3,1_]_bypass/model_best.pth.tar'
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = chresmax_v3_2_abs(num_classes=1000, big_size=322, small_size=322, in_chans=3, 
                 ip_scale_bands=3, classifier_input_size=18432, pyramid=False,
                 bypass=True, main_route=False, validation=True,
                 c_scoring='v2'      
    ).to(device).eval()
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    return model


def load_chresmax_v3_2(device='cuda'):
    """Load CHResMax v3.2 model"""
    from timm.models.RESMAX import chresmax_v3_2
    
    checkpoint_path = '/oscar/data/tserre/xyu110/pytorch-output/train/0/final_versions/ip_3_chresmax_v3_2_gpu_8_cl_0.1_ip_3_322_322_18432_c1[_6,3,1_]_bypass/model_best.pth.tar'
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = chresmax_v3_2(num_classes=1000, big_size=322, small_size=322, in_chans=3, 
                 ip_scale_bands=3, classifier_input_size=18432, pyramid=False,
                 bypass=True, main_route=False, validation=True,
                 c_scoring='v2'      
    ).to(device).eval()
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    return model


def load_alexnet_with_aug(device='cuda'):
    """Load AlexNet trained with augmentation"""
    from timm.models.alexnet import alexnet
    
    checkpoint_path = "/oscar/data/tserre/xyu110/pytorch-output/train/0/baseline_w_aug/ip_0_alexnet_gpu_2_cl_0_ip_3_227_227_0_c1[_6,3,1_]_scale_0.08/model_best.pth.tar"
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = alexnet(channel_size=227).to(device).eval()
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    return model


def load_alexnet_without_aug(device='cuda'):
    """Load AlexNet trained without augmentation"""
    from timm.models.alexnet import alexnet
    
    checkpoint_path = "/oscar/data/tserre/xyu110/pytorch-output/train/0/baseline_wo_aug/ip_0_alexnet_gpu_2_cl_0_ip_3_227_227_0_c1[_6,3,1_]/model_best.pth.tar"
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = alexnet(channel_size=227).to(device).eval()
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    return model


def load_resnet_without_aug(device='cuda'):
    """Load ResNet18 trained without augmentation"""
    from timm.models.resnet import resnet18
    
    checkpoint_path = "/oscar/data/tserre/xyu110/pytorch-output/train/0/baseline_wo_aug/ip_0_resnet18_gpu_8_cl_0_ip_3_227_227_512_c1[_6,3,1_]/model_best.pth.tar"
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = resnet18(channel_size=227).to(device).eval()
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    return model


def load_resnet_with_aug(device='cuda'):
    """Load ResNet18 trained with augmentation"""
    from timm.models.resnet import resnet18
    
    checkpoint_path = "/oscar/data/tserre/xyu110/pytorch-output/train/0/baseline_w_aug/ip_0_resnet18_gpu_8_cl_0_ip_3_227_227_512_c1[_6,3,1_]_scale_0.08/model_best.pth.tar"
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = resnet18(channel_size=227).to(device).eval()
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    return model


def load_alexnet_timm(device='cuda'):
    """Load AlexNet with timm-pretrained weights."""
    from timm.models.alexnet import alexnet

    model = alexnet(pretrained=True).to(device).eval()
    return model


def load_resnet18_timm(device='cuda'):
    """Load ResNet-18 torchvision weights through timm registry."""
    from timm.models import create_model

    model = create_model(
        'resnet18.tv_in1k',
        pretrained=True,
        num_classes=1000,
        in_chans=3,
        global_pool='avg',
        scriptable=False,
    ).to(device).eval()
    return model


def load_hmax_v3_adj(device='cuda'):
    """Load HMAX v3 adjusted model"""
    from timm.models.RESMAX import hmax_v3_adj
    
    checkpoint_path = '/oscar/data/tserre/xyu110/pytorch-output/train/0/final_versions/ip_3_hmax_v3_adj_gpu_8_cl_0.1_ip_3_322_322_18432_c1[_6,3,1_]_bypass/model_best.pth.tar'
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = hmax_v3_adj().to(device).eval()
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    return model


def load_chmax(device='cuda'):
    kwargs = {
        'ip_scale_bands': 18,
        'classifier_input_size': 4096,
        'bypass': True,
        'c_debug': False,
    }
    # "/oscar/data/tserre/xyu110/pytorch-output/train/0/mnist/ip_18_hmax_old_gpu_1_cl_0.5_ip_3_224_224_0000_c1[_6,3,1_]_bypass_1/model_best.pth.tar",
    # /oscar/home/npant1/data/npant1/HMAX-epoch=59-val_acc1=99.36899038461539-val_loss=0.029037245774629693.ckpt
    model = create_model(
        'hmax_old',
        pretrained="/oscar/data/tserre/xyu110/pytorch-output/train/0/mnist/ip_18_hmax_old_gpu_1_cl_0.5_ip_3_224_224_0000_c1[_6,3,1_]_bypass_1/model_best.pth.tar",
        num_classes=10,
        in_chans=3,
        global_pool=None,
        scriptable=False,
        **kwargs
    )
    
    # Set the critical attributes that your friend identified
    model.model_pre.base_scale = 224
    model.model_pre.ip_scales = 18
    
    return model.to(device).eval()


def load_vit_base(device='cuda'):
    """Load Vision Transformer Base model with ImageNet pretrained weights
    
    Uses ViT-Base (ViT-B/16) architecture with patch size 16 and 224x224 input.
    Pretrained weights are from ImageNet-1k, fine-tuned from ImageNet-21k.
    
    Args:
        device (str): Device to load the model on ('cuda' or 'cpu')
    
    Returns:
        VisionTransformer: ViT-Base model in evaluation mode
    """
    from timm.models.vision_transformer import vit_base_patch16_224
    
    # Load pretrained ViT-Base model with ImageNet weights
    model = vit_base_patch16_224(pretrained=True).to(device).eval()
    return model
