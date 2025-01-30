import os
import torch
import pandas as pd
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.utils import make_grid


class ScaledImagenetDataset(Dataset):
    """Scaled ImageNet dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (str): Path to CSV file containing metadata.
            root_dir (str): Root directory containing all images.
            transform (callable, optional): Transformations applied to samples.
        """
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Extract row values using iloc and idx
        wordnet_id = self.data.iloc[idx, 0]
        image_id = self.data.iloc[idx, 1]
        img_relative_path = self.data.iloc[idx, 2]
        mask_path = self.data.iloc[idx, 3]
        class_name = self.data.iloc[idx, 4]
        scale_band = int(self.data.iloc[idx, 5])
        relative_center_x = float(self.data.iloc[idx, 6])
        relative_center_y = float(self.data.iloc[idx, 7])

        # Construct full image path using root_dir
        img_name = os.path.join(self.root_dir, img_relative_path)

        # Verify file existence
        if not os.path.exists(img_name):
            raise FileNotFoundError(f"Image file does not exist: {img_name}")
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Mask file does not exist: {mask_path}")

        # Load image
        image = Image.open(img_name).convert("RGB")

        # Load mask
        mask_data = np.load(mask_path)
        mask = mask_data[mask_data.files[0]]

        # Convert relative centers to tensor
        relative_center = torch.tensor([relative_center_x, relative_center_y], dtype=torch.float32)

        if self.transform:
            image = self.transform(image)
            mask = Image.fromarray(mask).convert("L")
            mask = transforms.Resize((image.shape[1], image.shape[2]))(mask)
            mask = torch.tensor(np.array(mask), dtype=torch.float32)
            mask = torch.stack([mask] * 3, dim=0)

        sample = {
            'image': image,
            'mask': mask,
            'scale_band': scale_band,
            'relative_center': relative_center,
            'class_name': class_name
        }
        return sample


# Define transformations
transform_pipeline = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.CenterCrop(224)
])

# Define dataset and DataLoader
csv_file = "/cifs/data/tserre_lrs/projects/projects/prj_hmax_masks/HMAX/SAM_Imagenet/sam2/imagenet_centers.csv"
root_dir = "/gpfs/data/shared/imagenet/ILSVRC2012/train/"

dataset = ScaledImagenetDataset(
    csv_file=csv_file,
    root_dir=root_dir,
    transform=transform_pipeline
)

dataloader = DataLoader(dataset, batch_size=10, shuffle=True, num_workers=2)


# Helper function to visualize a batch of images, masks, and centers
def show_batch(sample_batched, save_path=None):
    images_batch = sample_batched['image']
    masks_batch = sample_batched['mask']
    centers_batch = sample_batched['relative_center']
    scale_bands_batch = sample_batched['scale_band']
    class_names_batch = sample_batched['class_name']
    batch_size = len(images_batch)
    grid_border_size = 2

    # Create grid of images
    images_grid = make_grid(images_batch, nrow=batch_size, padding=grid_border_size)

    # Create grid of masks
    masks_grid = make_grid(masks_batch, nrow=batch_size, padding=grid_border_size)

    # Combine images and masks into one visualization
    fig, axs = plt.subplots(2, 1, figsize=(15, 10))
    axs[0].imshow(images_grid.numpy().transpose((1, 2, 0)))
    axs[0].set_title("Images with Centers, Classes, and Scale Bands")
    axs[0].axis("off")

    axs[1].imshow(masks_grid.numpy().transpose((1, 2, 0)))
    axs[1].set_title("Masks")
    axs[1].axis("off")

    crop_size = 224
    for i in range(batch_size):
        center_x, center_y = centers_batch[i].numpy() * crop_size
        scale_band = scale_bands_batch[i].item()
        class_name = class_names_batch[i]
        grid_offset_x = i * (crop_size + grid_border_size)
        grid_offset_y = grid_border_size

        axs[0].scatter(center_x + grid_offset_x, center_y + grid_offset_y, s=20, marker='o', c='r')
        axs[0].text(grid_offset_x + crop_size / 2, grid_offset_y - 10, f"Scale: {scale_band}\nClass: {class_name}",
                    ha='center', color='white', fontsize=8, bbox=dict(facecolor='black', alpha=0.5))

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Figure saved to {save_path}")
    else:
        plt.show()


"""
for i_batch, sample_batched in enumerate(dataloader):
    print(f"Loaded batch {i_batch}")
    if i_batch == 0:
        output_dir = "output_figures"
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, f"batch_{i_batch}.png")
        show_batch(sample_batched, save_path=save_path)
        break
"""
