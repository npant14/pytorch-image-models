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


class ScaledImagenetDataset(Dataset):
    def __init__(self, csv_file, root_dir, mask_lookup_json, transform=None, crop_size=322):
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.crop_size = crop_size
        
        with open(mask_lookup_json, 'r') as f:
            self.mask_lookup = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_file = self.data.iloc[idx, 0]  # Image File
        scale_band = int(self.data.iloc[idx, 9])  # Scale Band
        center_x = float(self.data.iloc[idx, 7])  # Cropped center X
        center_y = float(self.data.iloc[idx, 8])  # Cropped center Y
        
        if img_file not in self.mask_lookup:
            raise FileNotFoundError(f"Image file {img_file} not found in mask lookup JSON.")
        
        img_path = self.mask_lookup[img_file]["image_path"]
        mask_path = self.mask_lookup[img_file]["mask_path"]

        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image file does not exist: {img_path}")
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Mask file does not exist: {mask_path}")

        image = Image.open(img_path).convert("RGB")
        mask_data = np.load(mask_path)
        mask = mask_data[mask_data.files[0]]

        if self.transform:
            image = self.transform(image)
            mask = Image.fromarray(mask).convert("L")
            mask = transforms.Resize((486, 486))(mask)
            mask = torch.tensor(np.array(mask), dtype=torch.float32)
            mask = torch.stack([mask] * 3, dim=0)
            mask = transforms.CenterCrop(322)(mask)

        center = torch.tensor([center_x, center_y], dtype=torch.float32)

        sample = {
            'image': image,
            'mask': mask,
            'scale_band': scale_band,
            'center': center
        }
        return sample


def show_batch(sample_batched, save_path=None):
    images_batch = sample_batched['image']
    masks_batch = sample_batched['mask']
    centers_batch = sample_batched['center']
    scale_bands_batch = sample_batched['scale_band']
    batch_size = len(images_batch)
    grid_border_size = 2

    images_grid = make_grid(images_batch, nrow=batch_size, padding=grid_border_size)
    masks_grid = make_grid(masks_batch, nrow=batch_size, padding=grid_border_size)

    fig, axs = plt.subplots(2, 1, figsize=(15, 10))
    axs[0].imshow(images_grid.numpy().transpose((1, 2, 0)))
    axs[0].set_title("Images with Centers and Scale Bands")
    axs[0].axis("off")

    axs[1].imshow(masks_grid.numpy().transpose((1, 2, 0)))
    axs[1].set_title("Masks")
    axs[1].axis("off")

    crop_size = 322
    for i in range(batch_size):
        center_x, center_y = centers_batch[i].numpy()
        scale_band = scale_bands_batch[i].item()
        grid_offset_x = i * (crop_size + grid_border_size)
        grid_offset_y = grid_border_size

        axs[0].scatter(
            grid_offset_x + center_x,
            grid_offset_y + center_y,
            s=20, marker='o', c='r'
        )
        axs[0].text(
            grid_offset_x + crop_size/2,
            grid_offset_y - 10,
            f"Scale: {scale_band}",
            ha='center', color='white', fontsize=8,
            bbox=dict(facecolor='black', alpha=0.5)
        )

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Figure saved to {save_path}")
    else:
        plt.show()


transform_pipeline = transforms.Compose([
    transforms.Resize((486, 486)),
    transforms.ToTensor(),
    transforms.CenterCrop(322)
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

dataloader = DataLoader(dataset, batch_size=100, shuffle=True, num_workers=2)

"""

for i_batch, sample_batched in enumerate(dataloader):
    if i_batch == 0:
        output_dir = "/cifs/data/tserre_lrs/projects/projects/prj_hmax_masks/HMAX/SAM_Imagenet/sam2/output_figures"
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, f"batch_{i_batch}.png")
        show_batch(sample_batched, save_path=save_path)
        break

"""
