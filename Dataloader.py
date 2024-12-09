import os
import glob
import numpy as np
import pandas as pd
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.ndimage import label


class ImageNetCustomDataset(Dataset):
    def __init__(self, img_root, csv_file, label_map_file, mask_directory, img_dim=(224, 224)):
        """
        Custom Dataset for ImageNet images with metadata and masks.

        Args:
        - img_root (str): Root directory containing image subfolders.
        - csv_file (str): CSV file with image metadata and scale bands.
        - label_map_file (str): File mapping WordNet IDs to class labels.
        - mask_directory (str): Directory containing .npz mask files.
        - img_dim (tuple): Target dimensions for resizing images (height, width).
        """
        self.img_root = img_root
        self.img_dim = img_dim
        self.mask_directory = mask_directory

        # Load mapping of class names to WordNet IDs
        self.class_to_wordnet = self._load_class_map(label_map_file)

        # Load metadata from the CSV file
        self.data = pd.read_csv(csv_file)

        # Normalize class names in the CSV
        self.data['Class'] = self.data['Class'].str.replace("_", " ").str.lower()

        # Dynamically load image paths and their WordNet IDs
        self.img_files = self._load_image_paths()

    @staticmethod
    def _load_class_map(label_map_file):
        """Load mapping of normalized class names to WordNet IDs."""
        class_map = {}
        with open(label_map_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 3:
                    wordnet_id = parts[0]
                    class_name = parts[2].replace("_", " ").lower()
                    class_map[class_name] = wordnet_id
        return class_map

    def _load_image_paths(self):
        """Dynamically find all image paths and associate them with WordNet IDs."""
        img_files = []
        for folder_path in glob.glob(os.path.join(self.img_root, "*")):
            if os.path.isdir(folder_path):
                wordnet_id = os.path.basename(folder_path)
                image_paths = glob.glob(os.path.join(folder_path, "*.jpeg"))
                if not image_paths:
                    print(f"Warning: No images found in {folder_path}.")
                img_files.extend((img_path, wordnet_id) for img_path in image_paths)
        if not img_files:
            raise ValueError(f"No images found in {self.img_root}. Check folder structure.")
        print(f"Loaded {len(img_files)} images from {self.img_root}.")
        return img_files

    @staticmethod
    def calculate_mask_centers(mask):
        """Calculate the centers of all objects in the mask."""
        centers = []
        labeled_mask, num_objects = label(mask)
        for object_id in range(1, num_objects + 1):
            object_mask = labeled_mask == object_id
            rows, cols = np.where(object_mask)
            if len(rows) == 0 or len(cols) == 0:
                continue

            # Get bounding box and calculate diagonal center
            top_left = (rows.min(), cols.min())
            bottom_right = (rows.max(), cols.max())
            row_center = (top_left[0] + bottom_right[0]) // 2
            col_center = (top_left[1] + bottom_right[1]) // 2

            centers.append((row_center, col_center))
        return centers

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        """
        Fetch an image, its associated metadata, and mask centers.

        Args:
        - idx (int): Index of the image to retrieve.

        Returns:
        - img_tensor (torch.Tensor): Preprocessed image tensor (C x H x W).
        - wordnet_id (str): WordNet ID associated with the image.
        - scale_band_tensor (torch.Tensor): Scale band metadata as a tensor.
        - mask_centers (list of tuples): Centers of the objects in the mask.
        """
        img_path, wordnet_id = self.img_files[idx]

        # Find the normalized class name for the WordNet ID
        class_name = next((k for k, v in self.class_to_wordnet.items() if v == wordnet_id), None)
        if class_name is None:
            raise ValueError(f"WordNet ID '{wordnet_id}' does not map to any class.")

        # Retrieve the scale band from the normalized CSV metadata
        row = self.data[self.data['Class'] == class_name]
        if row.empty:
            raise ValueError(f"No metadata found for class '{class_name}'.")
        scale_band = int(row.iloc[0]['Scale Band'])

        # Load and preprocess the image
        img = self._load_image(img_path)

        # Load the mask and calculate centers
        mask_file = os.path.join(self.mask_directory, f"{os.path.basename(img_path).split('.')[0]}.npz")
        if not os.path.exists(mask_file):
            mask_centers = []  # No mask file found
        else:
            mask_data = np.load(mask_file)
            if 'masks' in mask_data:
                mask = mask_data['masks']
                mask_centers = self.calculate_mask_centers(mask)
            else:
                mask_centers = []

        # Convert scale band to a tensor
        scale_band_tensor = torch.tensor(scale_band, dtype=torch.long)

        return img, wordnet_id, scale_band_tensor, mask_centers

    def _load_image(self, img_path):
        """Load and preprocess an image from the given path."""
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        img = cv2.resize(img, self.img_dim)
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float()
        return img_tensor


# Define paths
img_root = "/gpfs/data/shared/imagenet/ILSVRC2012/train/"
csv_file = "/oscar/scratch/vnema/foreground_proportions.csv"
label_map_file = "/users/vnema/HMAX/SAM_Imagenet/sam2/wordnetids_to_labels.txt"
mask_directory = "/users/vnema/scratch/ImageNet_Mask_npz"

# Initialize dataset and DataLoader
try:
    dataset = ImageNetCustomDataset(img_root, csv_file, label_map_file, mask_directory)
    print(f"Dataset initialized with {len(dataset)} images.")
except ValueError as e:
    print(f"Error initializing dataset: {e}")
    exit(1)

dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

# Example usage: Iterate through DataLoader
try:
    for batch_idx, (images, wordnet_ids, scale_bands, mask_centers) in enumerate(dataloader):
        print(f"Batch {batch_idx}:")
        print(f"Images shape: {images.shape}")
        print(f"WordNet IDs: {wordnet_ids}")
        print(f"Scale Bands: {scale_bands}")
        print(f"Mask Centers: {mask_centers}")
        break
except Exception as e:
    print(f"Error during DataLoader iteration: {e}")
