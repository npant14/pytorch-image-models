
import torch

#import tensorflow as tf
import numpy as np
from xplique.attributions import Saliency
import numpy as np
import torch
import torchvision
from torchvision import transforms
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

from harmonization.common import load_clickme_val
from harmonization.evaluation import evaluate_clickme

import os
import torch


import sys
sys.path.append("/files22_lrsresearch/CLPS_Serre_Lab/projects/prj_hmax_masks/pytorch-image-models-main/timm")

from models.RESMAX import resmax_bypass, RESMAX_Bypass
from models.ALEXMAX3 import ALEXMAX_v3, alexmax_v3


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_ResMax_bypass():
    
    """
    Loads the RESMAX_Bypass PyTorch model with pretrained weights.

    Returns
    -------
    model : torch.nn.Module
        RESMAX_Bypass model loaded with pretrained weights.
    """
    checkpoint_path = "/cifs/data/tserre_lrs/projects/prj_hmax/models/debug_resmax_bypass_concat_gpu_8_cl_0_ip_3_322_322_18432_c1[_6,3,1_]/model_best.pth.tar"
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model = resmax_bypass(num_classes=1000, big_size=322, small_size=227,classifier_input_size=18432)  # make sure this matches the architecture used in training
    model.load_state_dict(checkpoint['state_dict'])

    return model

from models.ALEXMAX3 import ALEXMAX_v3_1, alexmax_v3_1

def load_ALEXMAX_v3_1():
    checkpoint_path = "/cifs/data/tserre_lrs/projects/prj_hmax/models/resize2_alexmax_v3.1_cl_1_ip_{3}_322_12544/model_best.pth.tar"
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model = alexmax_v3_1(num_classes=1000, classifier_input_size=12544).to(device)
    missing, unexpected = model.load_state_dict(checkpoint['state_dict'], strict=False)
    print("Missing keys:", missing)
    print("Unexpected keys:", unexpected)

    return model



model = load_ALEXMAX_v3_1()
model = model.to(device).eval()  

from harmonization.common import load_clickme_train

clickme_ds = load_clickme_train(batch_size = 128)

"""
for images, heatmaps, labels in clickme_ds:
    print(images.shape) # (128, 224, 224, 3)
    print(heatmaps.shape) # (128, 224, 224, 1)
    print(labels.shape) # (128, 1000)
    """

config = resolve_data_config({}, model=model)
transform = create_transform(**config)
model_preprocess = transforms.Compose([transforms.ToPILImage(), create_transform(**config)])

def torch_explainer(xbatch, ybatch):
    # Convert to tensor and preprocess
    xbatch = torch.stack([model_preprocess(x) for x in xbatch.numpy().astype(np.uint8)])
    ybatch = torch.Tensor(ybatch.numpy())

    # Move to device
    xbatch = xbatch.to(device)
    ybatch = ybatch.to(device)

    xbatch.requires_grad_()

    out = model(xbatch)
    output = torch.sum(out * ybatch)
    output.backward()

    saliency, _ = torch.max(xbatch.grad.data.abs(), dim=1)
    return saliency.cpu().numpy()  # return to CPU as numpy array


clickme_dataset = load_clickme_val(batch_size = 128)

scores = evaluate_clickme(model,
                          explainer=torch_explainer,
                          clickme_val_dataset=clickme_dataset.take(5))
print(scores['alignment_score'])



