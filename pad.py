import random
import torch
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

def visualize_dataloader_samples(loader, target_size, num_images=16, save_path='dataloader_samples.png'):
    """
    Visualize and save samples from dataloader
    Args:
        loader: PyTorch dataloader
        num_images (int): Number of images to visualize
        save_path (str): Path to save the visualization
    """
    # Get a batch of images
    images, _ = next(iter(loader))
    images = pad_batch_random(images, target_size)
    
    # Select only the number of images we want to display
    images = images[:num_images]
    
    # Create a grid of images
    grid = vutils.make_grid(images, nrow=4, padding=2, normalize=True)
    
    # Convert to numpy and transpose for plotting
    grid = grid.cpu().numpy().transpose((1, 2, 0))
    
    # Create figure and display
    plt.figure(figsize=(15, 15))
    plt.axis('off')
    plt.imshow(grid)
    
    # Save the figure
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    
    print(f"Visualization saved to {save_path}")