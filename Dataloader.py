
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch
import matplotlib.pyplot as plt

class ScaledImagenetDataset(Dataset):
    
    def __init__(self, csv_file, root_dir, transform = None):
        """
        csv_file (str): Path to csv file with all foreground proportions and center of masks 
        root_dir (str): Directory of all images 
        
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
        img_name = os.path.join(self.root_dir, wordnet_id, f"{wordnet_id}_{image_id}.JPEG")  # Assuming images have '.JPEG' extension

        # Verify that the path points to a valid file
        if not os.path.isfile(img_name):
            raise FileNotFoundError(f"File does not exist or is not an image: {img_name}")

        # Load the image
        try:
            image = io.imread(img_name)  # Load the image
        except Exception as e:
            raise ValueError(f"Failed to load image {img_name}: {e}")

        # Extract scale band and mask centers
        scale_band = self.masks.iloc[idx]['Scale Band']
        mask_centers = self.masks.iloc[idx, 4:].values.astype(float)  # Assuming 'Center Row' and 'Center Column' start from column index 4
        class_label = self.masks.iloc[idx, 0]
        # Create the sample dictionary
        sample = {'image': image, 'class': class_label, 'scale_band': scale_band, 'mask_centers': mask_centers}

        if self.transform:
            sample = self.transform(sample)

        return sample

    
img_root = "/gpfs/data/shared/imagenet/ILSVRC2012/train/"  # Root path of ImageNet train data
csv_file = "/oscar/scratch/vnema/foreground_proportions_all.csv"  # Path to the CSV file
label_map_file = "/cifs/data/tserre_lrs/projects/vnema/SAM_Masks/wordnet_ids_to_class_labels.txt"  # Mapping file path
    
Image_dataset = ScaledImagenetDataset(csv_file, img_root)

fig = plt.figure()

for i, sample in enumerate(Image_dataset):
    print(i, sample['image'].shape, sample['class'], sample['mask_centers'], sample['scale_band'])

    ax = plt.subplot(1, 4, i + 1)
    plt.tight_layout()
    ax.set_title('Sample #{}'.format(i))
    ax.axis('off')
    #show_landmarks(**sample)

    if i == 3:
        plt.show()
        break
    


class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, mask_centers = sample['image'], sample['mask_centers']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))

        # Adjust mask centers based on scaling
        mask_centers = mask_centers * [new_w / w, new_h / h]

        return {'image': img, 'mask_centers': mask_centers}

    
class RandomCrop(object):
    """Randomly crops the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, mask_centers = sample['image'], sample['mask_centers']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h + 1)
        left = np.random.randint(0, w - new_w + 1)

        image = image[top: top + new_h,
                      left: left + new_w]

        # Adjust mask centers based on cropping
        mask_centers = mask_centers - [left, top]
        
        # Ensure mask centers are within bounds
        #mask_centers[:, 0] = np.clip(mask_centers[:], 0, new_h - 1)  # y-coordinates
        #mask_centers[:, 1] = np.clip(mask_centers[:, 1], 0, new_w - 1)  # x-coordinates

        return {'image': image, 'mask_centers': mask_centers}

class ScaledCrop(object):
    """Crop the image around the mask center, scaled based on the scale band.

    Args:
        crop_factor (float): Factor to scale the crop size around the mask center.
    """

    def __init__(self, crop_factor=2.0):
        """
        Args:
            crop_factor (float): The crop size is determined as scale_band * crop_factor.
        """
        self.crop_factor = crop_factor

    def __call__(self, sample):
        image, mask_centers, scale_band = sample['image'], sample['mask_centers'], sample['scale_band']

        h, w = image.shape[:2]

        # Ensure mask_centers is 2D
        if mask_centers.ndim == 1:
            mask_centers = mask_centers.reshape(-1, 2)

        if mask_centers.size == 0:  # If no mask centers are available
            return {'image': image, 'mask_centers': mask_centers, 'scale_band': scale_band}

        # Determine crop size based on scale_band
        crop_size = int(scale_band * self.crop_factor)

        # Iterate over all mask centers
        for center in mask_centers:
            center_x, center_y = int(center[1]), int(center[0])  # x, y coordinates

            # Calculate crop boundaries
            left = max(center_x - crop_size // 2, 0)
            top = max(center_y - crop_size // 2, 0)
            right = min(center_x + crop_size // 2, w)
            bottom = min(center_y + crop_size // 2, h)

            # Ensure crop size is valid
            crop_width = right - left
            crop_height = bottom - top

            if crop_width <= 0 or crop_height <= 0:
                raise ValueError(f"Invalid crop size: {crop_width}x{crop_height}")

            # Crop the image
            image = image[top:bottom, left:right]

            # Adjust mask centers relative to the cropped region
            mask_centers -= [left, top]

        return {'image': image, 'mask_centers': mask_centers, 'scale_band': scale_band}


class ToTensor(object):
    # Convert ndarrays in sample to Tensors 

    def __call__(self, sample):
        image, mask_centers = sample['image'], sample['mask_centers']

        # Swap color axis because
        # numpy image: H x W x C
        # torch image: C x H x W
        image = image.transpose((2, 0, 1))
        return {
            'image': torch.from_numpy(image).float(),
            'mask_centers': torch.from_numpy(mask_centers).float()
        }

transformed_dataset = ScaledImagenetDataset(
    csv_file=csv_file,  # Path to your CSV
    root_dir=img_root,  # Root directory for images
    transform=transforms.Compose([
        Rescale(256),  # Rescale to 256 while maintaining aspect ratio
        RandomCrop(224),  # Randomly crop to 224x224
        ToTensor()  # Convert to PyTorch tensors
    ])
)


transformed_dataset2 = ScaledImagenetDataset(
    csv_file=csv_file,  # Path to your CSV
    root_dir=img_root,  # Root directory for images
    transform=transforms.Compose([
        Rescale(256),           # Rescale the image
        ScaledCrop(crop_factor=2.0),  # TODO: Crop based on scale_band
        ToTensor()              # Convert to PyTorch tensors
    ])
)


dataloader = DataLoader(transformed_dataset, batch_size=10, shuffle=True, num_workers=2)

#dataloader2 = DataLoader(transformed_dataset2, batch_size=4, shuffle=True, num_workers=0)


# Helper function to show a batch of images with mask centers

def show_mask_centers_batch(sample_batched, save_path=None):
   
    images_batch, centers_batch = sample_batched['image'], sample_batched['mask_centers']
    batch_size = len(images_batch)
    grid_border_size = 2

    # Create a grid of images
    grid = utils.make_grid(images_batch, nrow=batch_size, padding=grid_border_size)
    plt.imshow(grid.numpy().transpose((1, 2, 0)))

    # Calculate the width and height of each image in the grid
    _, _, h, w = images_batch.size()

    for i in range(batch_size):

        centers = centers_batch[i].reshape(-1, 2)  # Reshape to (N, 2)
        
        # Ensure centers are within bounds
        centers[:, 0] = centers[:, 0].clamp(0, h - 1)  # Clamp y-coordinates
        centers[:, 1] = centers[:, 1].clamp(0, w - 1)  # Clamp x-coordinates

        # Adjust mask center coordinates relative to the grid position
        for center in centers:
            plt.scatter(
                center[1].item() + i * (w + grid_border_size),  # Adjust x-coordinate
                center[0].item() + grid_border_size,  # Adjust y-coordinate
                s=10, marker='o', c='r'
            )

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
    print(f"  Mask centers batch size: {sample_batched['mask_centers'].size()}")

    if i_batch == 2:
        save_path = os.path.join(output_dir, f"batch_{i_batch}.png")
        plt.figure(figsize=(10, 10))
        show_mask_centers_batch(sample_batched, save_path=save_path)
        break

