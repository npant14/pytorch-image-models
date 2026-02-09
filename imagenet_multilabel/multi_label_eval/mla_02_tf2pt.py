import tensorflow_datasets as tfds
import tensorflow as tf
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from scipy.ndimage import gaussian_filter
import os

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

# Transform tensorflow tensor to pytorch tensor
def tf2torch(t): # a batch of image tensors (N, H, W, 3)
    t = tf.cast(t, tf.float32).numpy()
    if len(t.shape) ==4 and t.shape[-1] in [1, 3]:
        t = torch.from_numpy(t.transpose(0, 3, 1, 2)) # torch.from_numpy(np_array.transpose(0, 3, 1, 2)) 
        return t # (N, 3, H, W)
    
    if len(t.shape) >= 1:
        return torch.from_numpy(t)
    else:
        return torch.tensor(t)

# Display torch/tf tensor
def show(img, p=False, smooth=False, **kwargs):

    try:
        img = img.detach().cpu()
    except:
        img = np.array(img)

    img = np.array(img, dtype=np.float32)

    # Remove the batch_size dim
    if len(img.shape) == 4:
        img = img[0]

    # check if channel first
    if img.shape[0] == 3 or img.shape[0] == 1:
        img = np.moveaxis(img, 0, -1)

    # normalize
    if img.max() > 1 or img.min() < 0:
        img -= img.min(); img /= img.max()

    # check if clip percentile
    if p is not False and len(img.shape) == 2:
        img = np.clip(img, np.percentile(img, p), np.percentile(img, 100-p))

    # check if smooth
    if smooth and img.shape[-1] == 1:
        img = gaussian_filter(img, smooth)

    ax.imshow(img, **kwargs)
    ax.axis('off')
    ax.grid(None)

ds = tfds.load('imagenet2012_multilabel', split='validation')
save_path = "../datasets/imagenet_multi_label/"

cnt = 0
for data in ds:
    correct_multi_labels = data['correct_multi_labels']
    file_name = data['file_name']
    image = data['image']
    is_problematic = data['is_problematic']
    original_label = data['original_label']
    unclear_multi_labels = data['unclear_multi_labels']
    wrong_multi_labels = data['wrong_multi_labels']

    if data['is_problematic'].numpy():
        continue

    cnt += 1

    # if cnt == 1:
    #     print(correct_multi_labels)
    #     print(file_name)
    #     print(image.dtype)
    #     print(original_label)
    #     print(unclear_multi_labels)
    #     print(wrong_multi_labels)

    image = tf2torch(image[None, :, :, :])[0]
    original_label = tf2torch(original_label)
    unclear_multi_labels = tf2torch(unclear_multi_labels)
    correct_multi_labels = tf2torch(correct_multi_labels)
    file_name = file_name.numpy().decode('utf-8')

    if cnt == 1:
        print(correct_multi_labels)
        print(file_name)
        print(image.dtype)
        print(original_label)
        print(unclear_multi_labels)

    instance = {
        'correct_multi_labels':correct_multi_labels,
        'image':image,
        'original_label':original_label,
        'unclear_multi_labels':unclear_multi_labels
    }

    # print(data)
    # print(image.max(), image.min())
    # break

    # Save the data to a *.pth file
    pth_path = os.path.join(save_path, file_name.split('.')[0] + ".pth")
    torch.save(instance, pth_path)
    
    print(pth_path + " is saved; %s \r" % (cnt), end=" ")
    
print("done")