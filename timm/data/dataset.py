""" Quick n Simple Image Folder, Tarfile based DataSet

Hacked together by / Copyright 2019, Ross Wightman
"""
import io
import logging
import os
import pandas as pd
import numpy as np
import sys
import json

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

# Get the directory of the current script
current_dir = os.path.dirname(__file__)
wordnet_to_label_txt = os.path.join(current_dir, '_info', 'wordnetids_to_labels.txt')
wordnet_to_label = {}
with open(wordnet_to_label_txt, 'r') as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) > 1:
            wordnet_to_label[parts[0]] = int(parts[1])-1


class ScaledImagenetDataset(data.Dataset):
    def __init__(
            self,
            root,
            csv_file=None,
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
        self.split = split
        
        if csv_file is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            csv_file = os.path.join(current_dir, '_info', 'foreground_proportions_with_rescaled_centers.csv')
        
        # Load the CSV data
        self.data = pd.read_csv(csv_file)
        _logger.info(f"Loaded {len(self.data)} samples from CSV file")
        
        # Get the list of files in the current split
        split_files = set(self.reader.filenames(basename=True))
        _logger.info(f"Found {len(split_files)} files in {split} split")
        
        # For validation set, we don't need to filter the CSV data
        # since we'll use default values for scale band and center
        if split != 'val':
            self.data = self.data[self.data['Image File'].isin(split_files)]
            _logger.info(f"Filtered CSV data to {len(self.data)} samples matching {split} split")
        
        _logger.info(f"Dataset contains {len(self.reader)} total samples")

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
                
        filename = self.reader.filename(index)
        fname = filename.split('/')[-1]
        
        # For validation set, use default values
        if self.split == 'val':
            scale_band = 3
            center_x = 0.5
            center_y = 0.5
        else:
            # For training set, get values from CSV
            if fname in self.data['Image File'].values:
                scale_band = int(self.data[self.data['Image File'] == fname]['Scale Band'].iloc[0])
                center_x = float(self.data[self.data['Image File'] == fname]['Cropped center X'].iloc[0])
                center_y = float(self.data[self.data['Image File'] == fname]['Cropped center Y'].iloc[0])
            else:
                scale_band = 3
                center_x = 0.5
                center_y = 0.5
        
        self._consecutive_errors = 0

        if self.input_img_mode and not self.load_bytes:
            img = img.convert(self.input_img_mode)
        if self.transform is not None:
            img = self.transform(img)

        if target is None:
            target = -1
        elif self.target_transform is not None:
            target = self.target_transform(target)

        center = torch.tensor([center_x, center_y], dtype=torch.float32)

        return img, target, scale_band, center

    def __len__(self):
        return len(self.reader)

    def filename(self, index, basename=False, absolute=False):
        return self.reader.filename(index, basename, absolute)

    def filenames(self, basename=False, absolute=False):
        return self.reader.filenames(basename, absolute)


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
