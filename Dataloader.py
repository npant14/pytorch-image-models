import os
import json
import torch
import pandas as pd
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

# Load WordNet ID to Class Label Mapping from text file
wordnet_to_label_txt = "/cifs/data/tserre_lrs/projects/projects/prj_hmax_masks/HMAX/SAM_Imagenet/EVF-SAM/wordnetids_to_labels.txt"
wordnet_to_label = {}
with open(wordnet_to_label_txt, 'r') as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) > 1:
            wordnet_to_label[parts[0]] = " ".join(parts[1:])

class ScaledImagenetDataset(Dataset):
    def __init__(self, csv_file, root_dir, mask_lookup_json, transform=None):
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        
        with open(mask_lookup_json, 'r') as f:
            self.mask_lookup = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_file = self.data.iloc[idx, 0]  # Image File
        scale_band = int(self.data.iloc[idx, 9])  # Scale Band
        center_x = float(self.data.iloc[idx, 7])  # Cropped center X
        center_y = float(self.data.iloc[idx, 8])  # Cropped center Y
        
        if img_file not in self.mask_lookup:
            raise FileNotFoundError(f"Image file {img_file} not found in mask lookup JSON.")
        
        img_path = self.mask_lookup[img_file]["image_path"]
        mask_path = self.mask_lookup[img_file]["mask_path"]
        
        wordnet_id = img_file.split('_')[0]  # Extract WordNet ID
        class_label = wordnet_to_label.get(wordnet_id, "Unknown")
        
        image = Image.open(img_path).convert("RGB")
        mask_data = np.load(mask_path)
        mask = mask_data[mask_data.files[0]]

        if self.transform:
            image = self.transform(image)
            mask = Image.fromarray(mask).convert("L")
            mask = transforms.Resize((322, 322))(mask)
            mask = torch.tensor(np.array(mask), dtype=torch.float32)
            mask = torch.stack([mask] * 3, dim=0)
        
        center = torch.tensor([center_x, center_y], dtype=torch.float32)
        
        sample = {
            'image': image,
            'mask': mask,
            'scale_band': scale_band,
            'center': center,
            'class_label': class_label
        }
        return sample

def show_batch(sample_batched, save_path=None):
    images_batch = sample_batched['image']
    masks_batch = sample_batched['mask']
    centers_batch = sample_batched['center']
    scale_bands_batch = sample_batched['scale_band']
    class_labels = sample_batched['class_label']
    batch_size = len(images_batch)

    fig, axs = plt.subplots(2, batch_size, figsize=(batch_size * 3, 6))
    
    for i in range(batch_size):
        image = images_batch[i].numpy().transpose((1, 2, 0))
        mask = masks_batch[i].numpy().transpose((1, 2, 0))
        center_x, center_y = centers_batch[i].numpy()
        scale_band = scale_bands_batch[i].item()
        class_label = class_labels[i]
        
        axs[0, i].imshow(image)
        axs[0, i].scatter(center_x, center_y, s=20, marker='o', c='r')
        axs[0, i].set_title(f"{class_label}\nScale: {scale_band}")
        axs[0, i].axis("off")
        
        axs[1, i].imshow(mask, cmap='gray')
        axs[1, i].axis("off")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Figure saved to {save_path}")
    else:
        plt.show()

transform_pipeline = transforms.Compose([
    transforms.Resize((322, 322)),
    transforms.ToTensor()
])

csv_file = "/oscar/scratch/vnema/foreground_proportions_with_rescaled_centers.csv"
root_dir = "/gpfs/data/tserre/npant1/ILSVRC/train"
mask_lookup_json = "/cifs/data/tserre_lrs/projects/projects/prj_hmax_masks/HMAX/SAM_Imagenet/sam2/image_to_mask_lookup.json"

dataset = ScaledImagenetDataset(
    csv_file=csv_file,
    root_dir=root_dir,
    mask_lookup_json=mask_lookup_json,
    transform=transform_pipeline
)

dataloader = DataLoader(dataset, batch_size=10, shuffle=True, num_workers=2)
"""
for i_batch, sample_batched in enumerate(dataloader):
    if i_batch == 0:
        output_dir = "/cifs/data/tserre_lrs/projects/projects/prj_hmax_masks/HMAX/SAM_Imagenet/sam2/output_figures"
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, f"batch_{i_batch}.png")
        show_batch(sample_batched, save_path=save_path)
        break
"""
