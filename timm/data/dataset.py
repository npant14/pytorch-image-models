""" Quick n Simple Image Folder, Tarfile based DataSet

Hacked together by / Copyright 2019, Ross Wightman
"""
import io
import logging
import os
import pandas as pd
import numpy as np
import sys


from typing import Optional

import torch
import torch.utils.data as data
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image

from .readers import create_reader

_logger = logging.getLogger(__name__)


_ERROR_RETRY = 50



def load_wordnet_to_numeric_mapping(txt_file_path: str) -> dict:
    """
    Reads a text file where each line contains a WordNet ID, a numeric value,
    and a class name, separated by whitespace. Returns a dictionary mapping
    each WordNet ID to the numeric value from the second column.

    Example input file line:
        n02119789 1 kit_fox

    Args:
        txt_file_path (str): Path to the text file.

    Returns:
        dict: A dictionary mapping from WordNet ID (str) to numeric value (int).
    """
    mapping = {}
    with open(txt_file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue  # Skip empty lines.
            parts = line.split()
            if len(parts) < 2:
                continue  # Skip lines that don't have at least two tokens.
            wordnet_id = parts[0]
            try:
                numeric_value = int(parts[1])
            except ValueError:
                # Skip this line or handle the error as needed.
                continue
            mapping[wordnet_id] = numeric_value
    return mapping

class ScaledImagenetDataset(Dataset):
    def __init__(self, csv_file, root_dir,root=None, transform=None, crop_size=224):
        """
        Args:
            csv_file (str): Path to CSV file containing metadata.
            root_dir (str): Root directory containing all images.
            transform (callable, optional): Transformations applied to samples.
            crop_size (int): The final crop size used in the transformation (default 224).
                             The image is first resized to a proportional size.
                             (Default ratio: 256/224)
        """
        
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.crop_size = crop_size
        # Compute the resize size so that the ratio crop_size:resize_size is the same as 224:256.
        self.resize_size = int(round(crop_size * (256 / 224)))
        self.class_path = os.path.join(root, 'imagenet_synset_raw.txt')
        self.class_map = load_wordnet_to_numeric_mapping(self.class_path)
        
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

        # Instead of assuming a fixed resize of 256x256, compute it from the crop_size.
        # Here, we assume that the transformation pipeline first resizes the image to (resize_size, resize_size)
        # and then applies a CenterCrop of (crop_size, crop_size).
        resized_h, resized_w = self.resize_size, self.resize_size  # e.g., 256 when crop_size is 224
        crop_h, crop_w = self.crop_size, self.crop_size           # e.g., 224

        # Convert relative centers to actual pixel coordinates based on the resized image.
        center_x = int(relative_center_x * resized_w)
        center_y = int(relative_center_y * resized_h)

        # Calculate the offset introduced by the CenterCrop.
        offset_x = (resized_w - crop_w) // 2
        offset_y = (resized_h - crop_h) // 2

        # Calculate new center coordinates for the cropped image.
        resized_center_x = center_x - offset_x
        resized_center_y = center_y - offset_y

        resized_center = torch.tensor([resized_center_x, resized_center_y], dtype=torch.float32)

        # Apply transformations if provided.
        if self.transform:
            image = self.transform(image)
            mask = Image.fromarray(mask).convert("L")
            # Resize mask to match the image dimensions (assumed to be (crop_size, crop_size)).
            mask = transforms.Resize((image.shape[1], image.shape[2]))(mask)
            mask = torch.tensor(np.array(mask), dtype=torch.float32)
            mask = torch.stack([mask] * 3, dim=0)

        sample = {
            'image': image,
            'mask': mask,
            'scale_band': scale_band,
            'resized_center': resized_center,
            'class_name': class_name,
            'target': self.class_map[wordnet_id]    
        }
        
        input = sample['image']#(sample['image'], sample['mask'], sample['scale_band'], sample['resized_center'])
        # make input a tensor
        
        target = sample['target']
        return input, target ,sample['scale_band']


class ImageDataset(data.Dataset):

    def __init__(
            self,
            root,
            reader=None,
            split='train',
            class_map=None,
            load_bytes=False,
            input_img_mode='RGB',
            transform=None,
            target_transform=None,
    ):
        if reader is None or isinstance(reader, str):
            reader = create_reader(
                reader or '',
                root=root,
                split=split,
                class_map=class_map
            )
        self.reader = reader
        self.load_bytes = load_bytes
        self.input_img_mode = input_img_mode
        self.transform = transform
        self.target_transform = target_transform
        self._consecutive_errors = 0

    def __getitem__(self, index):
        img, target = self.reader[index]

        try:
            img = img.read() if self.load_bytes else Image.open(img)
        except Exception as e:
            _logger.warning(f'Skipped sample (index {index}, file {self.reader.filename(index)}). {str(e)}')
            self._consecutive_errors += 1
            if self._consecutive_errors < _ERROR_RETRY:
                return self.__getitem__((index + 1) % len(self.reader))
            else:
                raise e
        self._consecutive_errors = 0

        if self.input_img_mode and not self.load_bytes:
            img = img.convert(self.input_img_mode)
        if self.transform is not None:
            img = self.transform(img)

        if target is None:
            target = -1
        elif self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.reader)

    def filename(self, index, basename=False, absolute=False):
        return self.reader.filename(index, basename, absolute)

    def filenames(self, basename=False, absolute=False):
        return self.reader.filenames(basename, absolute)


class IterableImageDataset(data.IterableDataset):

    def __init__(
            self,
            root,
            reader=None,
            split='train',
            class_map=None,
            is_training=False,
            batch_size=1,
            num_samples=None,
            seed=42,
            repeats=0,
            download=False,
            input_img_mode='RGB',
            input_key=None,
            target_key=None,
            transform=None,
            target_transform=None,
            max_steps=None,
    ):
        assert reader is not None
        if isinstance(reader, str):
            self.reader = create_reader(
                reader,
                root=root,
                split=split,
                class_map=class_map,
                is_training=is_training,
                batch_size=batch_size,
                num_samples=num_samples,
                seed=seed,
                repeats=repeats,
                download=download,
                input_img_mode=input_img_mode,
                input_key=input_key,
                target_key=target_key,
                max_steps=max_steps,
            )
        else:
            self.reader = reader
        self.transform = transform
        self.target_transform = target_transform
        self._consecutive_errors = 0

    def __iter__(self):
        for img, target in self.reader:
            if self.transform is not None:
                img = self.transform(img)
            if self.target_transform is not None:
                target = self.target_transform(target)
            yield img, target

    def __len__(self):
        if hasattr(self.reader, '__len__'):
            return len(self.reader)
        else:
            return 0

    def set_epoch(self, count):
        # TFDS and WDS need external epoch count for deterministic cross process shuffle
        if hasattr(self.reader, 'set_epoch'):
            self.reader.set_epoch(count)

    def set_loader_cfg(
            self,
            num_workers: Optional[int] = None,
    ):
        # TFDS and WDS readers need # workers for correct # samples estimate before loader processes created
        if hasattr(self.reader, 'set_loader_cfg'):
            self.reader.set_loader_cfg(num_workers=num_workers)

    def filename(self, index, basename=False, absolute=False):
        assert False, 'Filename lookup by index not supported, use filenames().'

    def filenames(self, basename=False, absolute=False):
        return self.reader.filenames(basename, absolute)


class AugMixDataset(torch.utils.data.Dataset):
    """Dataset wrapper to perform AugMix or other clean/augmentation mixes"""

    def __init__(self, dataset, num_splits=2):
        self.augmentation = None
        self.normalize = None
        self.dataset = dataset
        if self.dataset.transform is not None:
            self._set_transforms(self.dataset.transform)
        self.num_splits = num_splits

    def _set_transforms(self, x):
        assert isinstance(x, (list, tuple)) and len(x) == 3, 'Expecting a tuple/list of 3 transforms'
        self.dataset.transform = x[0]
        self.augmentation = x[1]
        self.normalize = x[2]

    @property
    def transform(self):
        return self.dataset.transform

    @transform.setter
    def transform(self, x):
        self._set_transforms(x)

    def _normalize(self, x):
        return x if self.normalize is None else self.normalize(x)

    def __getitem__(self, i):
        x, y = self.dataset[i]  # all splits share the same dataset base transform
        x_list = [self._normalize(x)]  # first split only normalizes (this is the 'clean' split)
        # run the full augmentation on the remaining splits
        for _ in range(self.num_splits - 1):
            x_list.append(self._normalize(self.augmentation(x)))
        return tuple(x_list), y

    def __len__(self):
        return len(self.dataset)
