
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

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

        # Create the sample dictionary
        sample = {'image': image, 'scale_band': scale_band, 'mask_centers': mask_centers}

        if self.transform:
            sample = self.transform(sample)

        return sample

    
img_root = "/gpfs/data/shared/imagenet/ILSVRC2012/train/"  # Root path of ImageNet train data
csv_file = "/oscar/scratch/vnema/foreground_proportions_all.csv"  # Path to the CSV file
label_map_file = "/cifs/data/tserre_lrs/projects/vnema/SAM_Masks/wordnet_ids_to_class_labels.txt"  # Mapping file path
    
Image_dataset = ScaledImagenetDataset(csv_file, img_root)

fig = plt.figure()

for i, sample in enumerate(Image_dataset):
    print(i, sample['image'].shape, sample['mask_centers'], sample['scale_band'])

    ax = plt.subplot(1, 4, i + 1)
    plt.tight_layout()
    ax.set_title('Sample #{}'.format(i))
    ax.axis('off')
    #show_landmarks(**sample)

    if i == 3:
        plt.show()
        break
