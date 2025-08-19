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

def load_images(datadir, curv, rot, size):
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
                img = Image.open(img_path).convert("RGB")  # Ensure RGB format
                img_tensor = transform(img)  # Convert to PyTorch tensor
                image_tensors.append((img_value, img_tensor))

    # Sort images by img number
    image_tensors.sort()

    # Extract sorted image tensors
    sorted_tensors = [img_tensor for _, img_tensor in image_tensors]
    print(f"Found {len(sorted_tensors)} images")
    return sorted_tensors

class Pasupathy():
    def __init__(self, model, outdir, device, data_dir, img_size=322, layer = "s3.layer.3.conv3"):
        self.model = model
        self.outdir = outdir
        self.device = device
        self.data_dir = data_dir
        self.img_size = img_size
        self.layer = layer
        print("setup Pasupathy experiment -- ready to run")

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
        activations = []
        print(f"Processing {len(images)} images at scale {scale}")
        
        # Create FeatureExtractor once instead of for each image
        layer_features = FeatureExtractor(model, [layer])
        
        for i, img in enumerate(images):
            if i % 10 == 0:  # Progress indicator
                print(f"Processing image {i+1}/{len(images)}")
                
            size = int(scale * img.shape[1]) 
            img = self.resize_image(img, size)
            features = layer_features(torch.unsqueeze(img, 0).to(self.device))
            tensor_feature = features[layer]
            
            # Extract center RF activations instead of max over all features
            # tensor_feature shape: (batch_size, channels, height, width)
            batch_size, channels, height, width = tensor_feature.shape
            
            # Get center coordinates
            center_h = height // 2
            center_w = width // 2
            
            # Extract center RF activations
            center_activations = tensor_feature[0, :, center_h, center_w]  # Shape: (channels,)
            
            # Take max over channels at center RF
            center_max_activation = torch.max(center_activations).item()
            activations.append(center_max_activation)

        return torch.argmax(torch.tensor(activations)).item()

    def run(self):
        slopes = np.zeros((0))
        total_combinations = 2 * 7  # 2 curv_sets * 7 rotations
        current_combination = 0
        
        for curv_set in [1,2]:
            for rot in range(1,8,1):
                current_combination += 1
                print(f"\n{'='*50}")
                print(f"Processing combination {current_combination}/{total_combinations}: curv={curv_set}, rot={rot}")
                print(f"{'='*50}")
                
                imgs = load_images(self.data_dir, curv_set, rot, self.img_size)
                print(f"Loaded {len(imgs)} images")
                
                selections = []
                for scale in [0.4, 0.6, 0.8, 1]:
                    print(f"\n--- Processing scale {scale} ---")
                    selection = self.pasupathy(self.model, imgs, scale, self.layer)
                    selections.append(selection)
                    print(f"Selection for scale {scale}: {selection}")
                
                # Fit line
                slope, _ = np.polyfit([0.4, 0.6, 0.8, 1], selections, 1)
                slopes = np.append(slopes, slope)
                print(f"Slope for this combination: {slope}")
                
                # Clean up GPU memory
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        final_mean = np.mean(slopes)
        print(f"\n{'='*50}")
        print(f"FINAL RESULT: Mean slope = {final_mean}")
        print(f"{'='*50}")
        return final_mean