import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils


class ScaledImagenetDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (str): Path to CSV file with all foreground proportions and centers.
            root_dir (str): Directory of all images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.masks = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.masks)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Construct the image path from 'WordNet ID' and 'Image ID'
        wordnet_id = self.masks.iloc[idx]['WordNet ID']
        image_id = self.masks.iloc[idx]['Image ID']
        img_name = os.path.join(self.root_dir, wordnet_id, f"{wordnet_id}_{image_id}.JPEG")

        # Verify the file exists
        if not os.path.isfile(img_name):
            raise FileNotFoundError(f"File does not exist or is not an image: {img_name}")

        # Load the image
        try:
            image = io.imread(img_name)
        except Exception as e:
            raise ValueError(f"Failed to load image {img_name}: {e}")

        # Extract scale band and relative centers
        scale_band = self.masks.iloc[idx]['Scale Band']
        rel_x = self.masks.iloc[idx]['Relative X']
        rel_y = self.masks.iloc[idx]['Relative Y']

        sample = {'image': image, 'scale_band': scale_band, 'relative_center': (rel_x, rel_y)}

        if self.transform:
            sample = self.transform(sample)

        return sample


class CenterCrop(object):
    def __init__(self, crop_size, resize_to=None):
        if isinstance(crop_size, int):
            self.crop_size = (crop_size, crop_size)
        else:
            self.crop_size = crop_size

        if resize_to is not None:
            if isinstance(resize_to, int):
                self.resize_to = (resize_to, resize_to)
            else:
                self.resize_to = resize_to
        else:
            self.resize_to = None

    def __call__(self, sample):
        image, rel_center = sample['image'], sample['relative_center']

        h, w = image.shape[:2]
        crop_h, crop_w = self.crop_size

        # Validate Relative X and Y
        if not (0 <= rel_center[0] <= 1 and 0 <= rel_center[1] <= 1):
            raise ValueError(f"Invalid relative center: {rel_center}")

        # Calculate absolute crop center using relative coordinates
        center_x = int(rel_center[0] * w)
        center_y = int(rel_center[1] * h)

        # Ensure crop box does not go out of bounds
        left = max(center_x - crop_w // 2, 0)
        top = max(center_y - crop_h // 2, 0)
        right = min(center_x + crop_w // 2, w)
        bottom = min(center_y + crop_h // 2, h)

        # Perform the crop
        image = image[top:bottom, left:right]

        # Resize the cropped image, if specified
        if self.resize_to is not None:
            image = transform.resize(image, self.resize_to, anti_aliasing=True)

        return {'image': image, 'scale_band': sample['scale_band'], 'relative_center': rel_center}



class ToTensor(object):
    def __call__(self, sample):
        image = sample['image']

        # Convert image to Tensor
        image = image.transpose((2, 0, 1)) if len(image.shape) == 3 else image[np.newaxis, :, :]

        return {
            'image': torch.from_numpy(image).float(),
            'scale_band': torch.tensor(sample['scale_band']).float(),
            'relative_center': torch.tensor(sample['relative_center']).float()
        }


# Define transformations
transformed_dataset = ScaledImagenetDataset(
    csv_file="/oscar/scratch/vnema/Combined_scale_centers.csv",
    root_dir="/gpfs/data/shared/imagenet/ILSVRC2012/train/",
    transform=transforms.Compose([
        CenterCrop(crop_size=224, resize_to=224),  # Use relative X, Y for cropping and resize
        ToTensor()
    ])
)

# Create DataLoader
dataloader = DataLoader(transformed_dataset, batch_size=10, shuffle=True, num_workers=2)


# Helper function to show a batch of images with relative centers
def show_batch(sample_batched, save_path=None):
    images_batch = sample_batched['image']
    centers_batch = sample_batched['relative_center']
    batch_size = len(images_batch)
    grid_border_size = 2

    grid = utils.make_grid(images_batch, nrow=batch_size, padding=grid_border_size)
    plt.imshow(grid.numpy().transpose((1, 2, 0)))

    for i in range(batch_size):
        center_x, center_y = centers_batch[i].numpy() * np.array([224, 224])  # Assuming 224x224 crop size
        plt.scatter(center_x + i * (224 + grid_border_size), center_y + grid_border_size, s=10, marker='o', c='r')

    plt.title('Batch from DataLoader')

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Figure saved to {save_path}")
    else:
        plt.show()


# Save or display a batch from DataLoader
output_dir = "output_figures"
os.makedirs(output_dir, exist_ok=True)

for i_batch, sample_batched in enumerate(dataloader):
    print(f"Batch {i_batch}:")
    print(f"  Image batch size: {sample_batched['image'].size()}")

    if i_batch == 2:
        save_path = os.path.join(output_dir, f"batch_{i_batch}.png")
        plt.figure(figsize=(10, 10))
        show_batch(sample_batched, save_path=save_path)
        break

