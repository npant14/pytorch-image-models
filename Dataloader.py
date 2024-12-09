import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import random
import glob
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class RandomScaleBandDataset(Dataset):
    def __init__(self):
        """
        Args:
            image_dir, csv_path, wordnet_to_labels_path, transform=None, with_replacement=True
            image_dir (str): Path to the directory containing subfolders with images.
            csv_path (str): Path to the CSV containing class and scale bands.
            wordnet_to_labels_path (str): Path to the WordNet ID to class name mapping file.
            transform (callable, optional): Transform to apply to images.
            with_replacement (bool, optional): Whether to sample with replacement. Default is True.
        """
        self.imgs_path= "/gpfs/data/shared/imagenet/ILSVRC2012/train/"
        file_list = golb.glob(self.imgs_path + "*")
        print(file_list)
        for class_path in file_list:
            class_name = class_path.split("/")[-1]
            for img_path in glob.glob(class_path + "/*.jpeg"):
                self.data.append([img_path, class_name])
        print(self.data)
        #self.class_map = {""}
        #self.image_dir = image_dir
        #self.transform = transform
        #self.with_replacement = with_replacement

        # Load WordNet ID to class name mapping
        print("Loading WordNet to class name mapping...")
        self.wordnet_to_class = self._load_wordnet_to_labels(wordnet_to_labels_path)

        # Load image metadata from the CSV
        print("Loading image metadata from CSV...")
        self.image_data = self._load_image_data(csv_path)

        # Resolve valid image paths and validate classes
        print("Finding valid images...")
        self.valid_image_files = self._find_valid_images()

        # Filter the DataFrame based on valid images
        print("Validating image data...")
        self.image_data = self._validate_image_data()

        # Expand the image data with full paths
        print("Expanding image data...")
        self.image_data = self.image_data.explode("Image Path").dropna(subset=["Image Path"])

        # Class-to-index mapping
        self.unique_classes = self.image_data["Class"].unique()
        self.class_to_index = {cls: idx for idx, cls in enumerate(self.unique_classes)}

        # Image paths and corresponding classes
        self.image_paths = self.image_data["Image Path"].tolist()
        self.image_classes = self.image_data["Class"].tolist()

        # Mapping for scale bands
        self.scale_band_mapping = self.image_data.groupby("Class")["Scale Band"].apply(list).to_dict()

    def _load_wordnet_to_labels(self, path):
        """Load WordNet to class name mapping."""
        wordnet_to_labels = {}
        try:
            with open(path, 'r') as file:
                for line in file:
                    parts = line.strip().split()
                    if len(parts) >= 3:
                        wordnet_id = parts[0].strip()
                        class_name = " ".join(parts[2:]).strip()
                        wordnet_to_labels[wordnet_id] = class_name
                    else:
                        print(f"Skipping invalid line: {line.strip()}")
        except Exception as e:
            raise ValueError(f"Error reading WordNet ID to class name mapping file: {e}")
        return wordnet_to_labels

    def _load_image_data(self, path):
        """Load and validate image data from CSV."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"CSV file not found: {path}")
        try:
            data = pd.read_csv(
                path,
                dtype={
                    "Class": str,
                    "Image ID": str,
                    "Foreground Proportion": float,
                    "Scale Band": int,
                },
            )
            print(f"Loaded CSV with columns: {data.columns.tolist()}")
            if "Class" not in data or "Scale Band" not in data:
                raise ValueError("Required columns ('Class', 'Scale Band') are missing in CSV.")
            return data
        except Exception as e:
            raise ValueError(f"Error reading CSV file: {e}")

    def _find_valid_images(self):
        """Find all valid images in the directory."""
        valid_image_files = {}
        for root, _, files in os.walk(self.image_dir):
            folder_name = os.path.basename(root).strip()
            class_name = self.wordnet_to_class.get(folder_name)
            if class_name:
                for file in files:
                    if file.lower().endswith((".jpg", ".jpeg", ".png")):
                        file_path = os.path.join(root, file)
                        valid_image_files.setdefault(class_name, []).append(file_path)
            else:
                print(f"Skipping folder without matching WordNet ID: {folder_name}")
        print(f"Found valid image files for classes: {list(valid_image_files.keys())}")
        return valid_image_files

    def _validate_image_data(self):
        """Filter the DataFrame for valid image paths."""
        if not set(self.image_data["Class"]).intersection(self.valid_image_files.keys()):
            raise ValueError("No valid classes found. Ensure WordNet IDs match the folder structure and CSV file.")
        self.image_data["Image Path"] = self.image_data["Class"].map(self.valid_image_files)
        return self.image_data

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        if self.with_replacement:
            idx = random.randint(0, len(self.image_paths) - 1)

        img_path = self.image_paths[idx]
        image_class = self.image_classes[idx]
        scale_band = random.choice(self.scale_band_mapping[image_class])

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            raise ValueError(f"Error loading image at {img_path}: {e}")

        if self.transform:
            image = self.transform(image)

        label = self.class_to_index[image_class]

        return {"image": image, "label": label, "scale_band": scale_band}


def create_random_dataloader(image_dir, csv_path, wordnet_to_labels_path, batch_size, with_replacement=True, shuffle=True, num_workers=4):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    dataset = RandomScaleBandDataset(
        image_dir=image_dir,
        csv_path=csv_path,
        wordnet_to_labels_path=wordnet_to_labels_path,
        transform=transform,
        with_replacement=with_replacement,
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


if __name__ == "__main__":
    image_dir = "/gpfs/data/shared/imagenet/ILSVRC2012/train/"
    csv_path = "/oscar/scratch/vnema/foreground_proportions.csv"
    wordnet_to_labels_path = "/users/vnema/HMAX/SAM_Imagenet/sam2/wordnetids_to_labels.txt"

    try:
        dataloader = create_random_dataloader(
            image_dir=image_dir,
            csv_path=csv_path,
            wordnet_to_labels_path=wordnet_to_labels_path,
            batch_size=32,
        )
        for batch_idx, batch in enumerate(dataloader):
            print(f"Batch {batch_idx + 1}:")
            print(f"  Image Batch Shape: {batch['image'].shape}")
            print(f"  Labels: {batch['label']}")
            print(f"  Scale Bands: {batch['scale_band']}")
            print("-" * 50)
            if batch_idx == 2:
                break
    except Exception as e:
        print(f"Error encountered: {e}")
