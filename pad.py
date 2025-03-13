import random
import torch
import numpy as np
from typing import Iterator
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import torchvision.utils as vutils
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt

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
            value=0
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
    padded_images = F.pad(images, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)

    return padded_images


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
    

class CenterResizeCropPad:
    def __init__(self, output_size=(227, 227), scale=160):
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
            
            # Apply padding
            transformed_img = F.pad(
                resized_img,
                (pad_left, pad_right, pad_top, pad_bottom),
                mode='constant',
                value=0
            )
        
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

def visualize_transforms(img, scales, target_size=(322, 322), save_path="transform_visualization.png"):
    """
    Visualize how an image looks after applying CenterResizeCropPad transform at different scales.
    
    Args:
        img: Input image tensor (B, C, H, W) - can be on CPU or GPU
        scales (list): List of scale values to visualize
        target_size (tuple): Output size for the transform
        save_path (str): Path to save the visualization
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    
    # Make sure we're working with a single image (not a batch)
    if img.dim() == 4:
        img_tensor = img[0]  # Take the first image if it's a batch
    else:
        img_tensor = img
    
    # Create figure for visualization
    n_scales = len(scales)
    fig, axes = plt.subplots(1, n_scales + 1, figsize=(4 * (n_scales + 1), 4))
    
    # Display original image - move to CPU first
    orig_img = img_tensor.cpu().permute(1, 2, 0).numpy()
    orig_img = np.clip(orig_img, 0, 1)
    axes[0].imshow(orig_img)
    axes[0].set_title(f'Original ({orig_img.shape[1]}x{orig_img.shape[0]})')
    axes[0].axis('off')
    
    # Apply transforms at different scales and visualize
    for i, scale in enumerate(scales):
        transform = CenterResizeCropPad(output_size=target_size, scale=scale)
        transformed_tensor = transform(img_tensor)
        
        # Convert tensor back to numpy for visualization - move to CPU first
        transformed_img = transformed_tensor.cpu().permute(1, 2, 0).numpy()
        # Clip values to be between 0 and 1
        transformed_img = np.clip(transformed_img, 0, 1)
        
        # Display transformed image
        axes[i+1].imshow(transformed_img)
        
        # Add title indicating whether it's padded or cropped
        if scale <= min(target_size):
            mode = "padded"
        else:
            mode = "cropped"
        axes[i+1].set_title(f'Scale {scale} ({mode})')
        axes[i+1].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Visualization saved to {save_path}")