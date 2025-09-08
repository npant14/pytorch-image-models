"""
runs an evaluation based on data from Pasupathy et. al. 2018(?) 
"""
from korean import FeatureExtractor
import torch
torch.manual_seed(1)
import numpy as np
from torchvision import transforms
from PIL import Image
import csv
import re
import os

# Default configuration constants
DEFAULT_CURV_SETS = [1, 2]
DEFAULT_ROTATIONS = list(range(1, 8))  # 1 to 7
DEFAULT_SCALES = [0.4, 0.6, 0.8, 1.0]
DEFAULT_IMG_SIZE = 322
DEFAULT_LAYER = "s3.layer.3.conv3"
DEFAULT_PROGRESS_INTERVAL = 10

def load_images(datadir, curv, rot, size):
    """
    Load images from directory.
    
    Args:
        datadir: Directory containing images
        curv: Curvature set to load
        rot: Rotation to load
        size: Target image size
    """
    pattern = re.compile(r"subplot_rot=(\d+)_curv=(\d+)_img=(\d+)\.png")
    # Define transformation to convert images to tensors
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),  # Converts image to tensor with shape (C, H, W)
    ])
    # List to store matched images as PyTorch tensors
    image_tensors = []

    print(f"Loading images from {datadir} for curv={curv}, rot={rot}")
    
    # Loop through all files in the folder
    for filename in os.listdir(datadir):
        match = pattern.search(filename)
        if match:
            rot_value = int(match.group(1))
            curv_value = int(match.group(2))
            img_value = int(match.group(3))

            if rot_value == rot and curv_value == curv:
                img_path = os.path.join(datadir, filename)
                try:
                    img = Image.open(img_path).convert("RGB")  # Ensure RGB format
                    img_tensor = transform(img)  # Convert to PyTorch tensor
                    image_tensors.append((img_value, img_tensor))
                except Exception as e:
                    print(f"Error processing {filename}: {e}")
                    continue

    # Sort images by img number
    image_tensors.sort()

    # Extract sorted image tensors
    sorted_tensors = [img_tensor for _, img_tensor in image_tensors]
    print(f"Found {len(sorted_tensors)} images")
    return sorted_tensors

class Pasupathy():
    def __init__(self, model, outdir, device, data_dir, img_size=DEFAULT_IMG_SIZE, layer=DEFAULT_LAYER, 
                 curv_sets=None, rotations=None, scales=None, progress_interval=DEFAULT_PROGRESS_INTERVAL):
        self.model = model
        self.outdir = outdir
        self.device = device
        self.data_dir = data_dir
        self.img_size = img_size
        self.layer = layer
        
        # Configurable parameters with sensible defaults
        self.curv_sets = curv_sets if curv_sets is not None else DEFAULT_CURV_SETS
        self.rotations = rotations if rotations is not None else DEFAULT_ROTATIONS
        self.scales = scales if scales is not None else DEFAULT_SCALES
        self.progress_interval = progress_interval
        
        print("setup Pasupathy experiment -- ready to run")
        print(f"Configuration: curv_sets={self.curv_sets}, rotations={self.rotations}, scales={self.scales}")

    def resize_image(self, image, size):
        image_size = image.shape[1]
        pil_image = transforms.ToPILImage()(image)
        resize = transforms.Compose([
                            transforms.Resize((size,size)),
                            transforms.Pad(((image_size - size)//2, (image_size - size)//2)),
                            transforms.Resize(self.img_size),
                            transforms.ToTensor()
                        ])
        new_img = resize(pil_image)

        return new_img


    def pasupathy(self, model, images, scale, layer):
        # Store all neuron responses for each image
        # activations will be a list of tensors, each with shape (channels,)
        activations = []
        print(f"Processing {len(images)} images at scale {scale}")
        
        # Create FeatureExtractor once instead of for each image
        layer_features = FeatureExtractor(model, [layer])
        
        for i, img in enumerate(images):
            if i % self.progress_interval == 0:  # Progress indicator
                print(f"Processing image {i+1}/{len(images)}")
                
            size = int(scale * img.shape[1]) 
            img = self.resize_image(img, size)
            
            # Process image with memory management
            with torch.no_grad():  # Disable gradients to save memory
                features = layer_features(torch.unsqueeze(img, 0).to(self.device))
                tensor_feature = features[layer]
                
                # Extract center RF activations instead of max over all features
                # Handle different tensor shapes for different layer types
                if len(tensor_feature.shape) == 4:
                    # Convolutional layers: (batch_size, channels, height, width)
                    batch_size, channels, height, width = tensor_feature.shape
                    
                    # Get center coordinates
                    center_h = height // 2
                    center_w = width // 2
                    
                    # Extract center RF activations for all channels (neurons)
                    center_activations = tensor_feature[0, :, center_h, center_w]  # Shape: (channels,)
                else:
                    # Skip layers like fc layers, batchnorm...
                    # Skip layers with unsupported tensor shapes (e.g., 1D, 3D, 5D+)
                    print(f"Warning: Skipping layer '{layer}' with unsupported tensor shape: {tensor_feature.shape}")
                    return None, None
                
                # Move to CPU immediately to save GPU memory
                center_activations = center_activations.detach().cpu()
                
                # Store the full neuron population response
                activations.append(center_activations)
                
                # Clear GPU tensors
                del tensor_feature, features

        # Stack all activations to get shape (num_images, num_channels)
        activations_tensor = torch.stack(activations)  # Shape: (num_images, num_channels)
        
        # Compute median response for each neuron across all presentations
        # activations_tensor shape: (num_images, num_channels)
        # median across dim=0 (images) gives us (num_channels,) - one value per neuron
        neuron_medians = torch.median(activations_tensor, dim=0)[0]  # Shape: (num_channels,)
        
        # Find the neuron with maximum median response
        best_neuron_idx = torch.argmax(neuron_medians).item()
        
        # Return both the best neuron index and the full vector for future analysis
        return best_neuron_idx, neuron_medians

    def run(self):
        slopes = np.zeros((0))
        total_combinations = len(self.curv_sets) * len(self.rotations)
        current_combination = 0
        
        # Store all neuron population data across all combinations
        all_neuron_populations = []
        
        for curv_set in self.curv_sets:
            for rot in self.rotations:
                current_combination += 1
                print(f"\n{'='*50}")
                print(f"Processing combination {current_combination}/{total_combinations}: curv={curv_set}, rot={rot}")
                print(f"{'='*50}")
                
                imgs = load_images(self.data_dir, curv_set, rot, self.img_size)
                print(f"Loaded {len(imgs)} images")
                
                selections = []
                neuron_populations = []  # Store the full neuron response vectors for this combination
                for scale in self.scales:
                    print(f"\n--- Processing scale {scale} ---")
                    selection, neuron_medians = self.pasupathy(self.model, imgs, scale, self.layer)
                    selections.append(selection)
                    neuron_populations.append(neuron_medians)
                    print(f"Selection for scale {scale}: {selection}")
                
                # Store this combination's neuron populations
                all_neuron_populations.append({
                    'curv_set': curv_set,
                    'rotation': rot,
                    'selections': selections,
                    'neuron_populations': neuron_populations
                })
                
                # Fit line using the configured scales
                slope, _ = np.polyfit(self.scales, selections, 1)
                slopes = np.append(slopes, slope)
                print(f"Slope for this combination: {slope}")
                
                # Clean up GPU memory
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        final_mean = np.mean(slopes)
        print(f"\n{'='*50}")
        print(f"FINAL RESULT: Mean slope = {final_mean}")
        print(f"{'='*50}")
        
        # Return both the final mean slope and all the neuron populations for future analysis
        # all_neuron_populations contains data for each combination (curv_set, rotation)
        # Each entry contains: curv_set, rotation, selections, neuron_populations
        # neuron_populations is a list of 4 tensors (one for each scale: 0.4, 0.6, 0.8, 1)
        # Each tensor has shape (num_channels,) representing median response per neuron
        
        # Also create a summary of the neuron population data for easier access
        neuron_summary = {
            'final_mean_slope': final_mean,
            'total_combinations': len(all_neuron_populations),
            'scales': self.scales,
            'curv_sets': self.curv_sets,
            'rotations': self.rotations,
            'combinations': all_neuron_populations
        }
        
        return final_mean, neuron_summary