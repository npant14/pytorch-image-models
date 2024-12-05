import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import random


class RandomScaleBandDataset(Dataset):
    def __init__(self, image_dir, csv_path, wordnet_to_labels_path, transform=None, with_replacement=True):
        """
        Args:
            image_dir (str): Path to the directory containing subfolders with images.
            csv_path (str): Path to the CSV containing class and scale bands.
            wordnet_to_labels_path (str): Path to the WordNet ID to class name mapping file.
            transform (callable, optional): Transform to apply to images.
            with_replacement (bool, optional): Whether to sample with replacement. Default is True.
        """
        self.image_dir = image_dir
        self.transform = transform
        self.with_replacement = with_replacement

        # Load WordNet ID to class name mapping from a .txt file
        try:
            wordnet_to_labels = {}
            with open(wordnet_to_labels_path, 'r') as file:
                for line in file:
                    parts = line.strip().split()
                    if len(parts) >= 3:
                        wordnet_id = parts[0].strip()
                        class_name = " ".join(parts[2:]).strip()
                        wordnet_to_labels[wordnet_id] = class_name
            self.wordnet_to_class = wordnet_to_labels
        except Exception as e:
            raise ValueError(f"Error reading WordNet ID to class name mapping file: {e}")

        # Load CSV file
        try:
            self.image_data = pd.read_csv(
                csv_path,
                dtype={
                    "Class": str,
                    "Image ID": str,
                    "Foreground Proportion": float,
                    "Scale Band": int,
                },
            )
        except Exception as e:
            raise ValueError(f"Error reading CSV file: {e}")

        # Validate required columns
        required_columns = {"Class", "Image ID", "Foreground Proportion", "Scale Band"}
        if not required_columns.issubset(self.image_data.columns):
            raise ValueError(f"CSV must contain columns: {required_columns}")

        # Resolve full paths to images based on their class names
        valid_image_files = {}
        for root, _, files in os.walk(self.image_dir):
            folder_name = os.path.basename(root).strip()
            class_name = self.wordnet_to_class.get(folder_name)
            if class_name:
                for file in files:
                    if file.lower().endswith((".jpg", ".jpeg", ".png")):
                        file_path = os.path.join(root, file)
                        valid_image_files.setdefault(class_name, []).append(file_path)

        # Filter CSV rows based on valid classes
        self.image_data = self.image_data[self.image_data["Class"].isin(valid_image_files.keys())]
        if self.image_data.empty:
            raise ValueError("No valid classes found. Ensure WordNet IDs match the folder structure and CSV file.")

        # Add image paths to the DataFrame
        self.image_data["Image Path"] = self.image_data.apply(
            lambda row: valid_image_files.get(row["Class"], []), axis=1
        )

        # Expand rows for each image path
        self.image_data = self.image_data.explode("Image Path").dropna(subset=["Image Path"])

        print(f"Number of valid images loaded: {len(self.image_data)}")

        # Group scale bands by class for random sampling
        self.scale_band_mapping = self.image_data.groupby("Class")["Scale Band"].apply(list).to_dict()

        # Unique classes and their mappings
        self.unique_classes = list(self.scale_band_mapping.keys())
        self.class_to_index = {cls: idx for idx, cls in enumerate(self.unique_classes)}

        # List of valid image paths and corresponding classes
        self.image_paths = self.image_data["Image Path"].tolist()
        self.image_classes = self.image_data["Class"].tolist()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Randomly sample an image
        if self.with_replacement:
            sampled_idx = random.randint(0, len(self.image_paths) - 1)
        else:
            sampled_idx = idx

        img_path = self.image_paths[sampled_idx]
        image_class = self.image_classes[sampled_idx]

        # Map class to a random scale band
        scale_band = random.choice(self.scale_band_mapping[image_class])

        # Load image
        image = Image.open(img_path).convert("RGB")

        # Apply transform
        if self.transform:
            image = self.transform(image)

        label = self.class_to_index[image_class]

        return {
            "image": image,
            "label": label,
            "scale_band": scale_band,
        }


def create_random_dataloader(
    image_dir, csv_path, wordnet_to_labels_path, batch_size, with_replacement=True, shuffle=True, num_workers=4
):
    """
    Creates a DataLoader for the RandomScaleBandDataset.
    Args:
        image_dir (str): Path to the directory containing subfolders with images.
        csv_path (str): Path to the CSV file.
        wordnet_to_labels_path (str): Path to the WordNet ID to class name mapping file.
        batch_size (int): Number of samples per batch.
        with_replacement (bool, optional): Whether to sample with replacement. Default is True.
        shuffle (bool, optional): Whether to shuffle the dataset. Default is True.
        num_workers (int, optional): Number of workers for data loading. Default is 4.
    Returns:
        DataLoader: DataLoader for the RandomScaleBandDataset.
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    dataset = RandomScaleBandDataset(
        image_dir, csv_path, wordnet_to_labels_path, transform=transform, with_replacement=with_replacement
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


# Example usage
if __name__ == "__main__":
    image_dir = "/gpfs/data/shared/imagenet/ILSVRC2012/train"  # Replace with actual path
    csv_path = "/oscar/scratch/vnema/foreground_proportions.csv"  # Replace with actual CSV file path
    wordnet_to_labels_path = "/users/vnema/HMAX/SAM_Imagenet/sam2/wordnetids_to_labels.txt"  # Replace with actual .txt file path

    try:
        dataloader = create_random_dataloader(
            image_dir=image_dir,
            csv_path=csv_path,
            wordnet_to_labels_path=wordnet_to_labels_path,
            batch_size=8,
            with_replacement=True,
        )
        # Iterate over the DataLoader and print batch information
        for batch_idx, batch in enumerate(dataloader):
            print(f"Batch {batch_idx + 1}:")
            print(f"  Image Batch Shape: {batch['image'].shape}")
            print(f"  Labels: {batch['label']}")
            print(f"  Scale Bands: {batch['scale_band']}")
            print("-" * 50)

            # Stop after testing a few batches
            if batch_idx == 2:  # Adjust the number of batches to test
                break         +

    except ValueError as e:
        print(f"Error: {e}")
