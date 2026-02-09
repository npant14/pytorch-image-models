import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms, models
from torch.utils.data import Subset, DataLoader, Dataset
from torchmetrics.regression import SpearmanCorrCoef

import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

import pandas as pd
import csv

import argparse
import glob
import os
import gc
import time
import warnings
warnings.filterwarnings("ignore")

from dino import DINOv1, DINOv2

def clear_unused_memo(device_name, needClearGC):
    if needClearGC:
        gc.collect()
    
    if device_name:
        with torch.cuda.device(device_name):
            torch.cuda.empty_cache()

def write_csv_all(record, path):
    header = ['model', 'multi_label_acc', 'time']
    file_exists = os.path.isfile(path)

    with open(path, mode='a+', newline='') as csv_file:
        writer = csv.writer(csv_file)
        if not file_exists:
            writer.writerow(header)
        writer.writerow(record)

def load_model(model_name, device):
    if model_name.endswith('_harmonized'):
        mn = model_name.split('_harmonized')[0]
        model = timm.create_model(mn, num_classes=1000, pretrained=False)
        root_path = '/media/data_cifs/projects/prj_pseudo_clickme/Checkpoints/for_adv'
        ckpt_path = os.path.join(root_path, f'{model_name}.pth.tar')
        checkpoint = torch.load(ckpt_path)
        model.load_state_dict(checkpoint['state_dict'])
        model = model.to(device)
    else:
        model = timm.create_model(model_name, pretrained=True, num_classes=1000).to(device)
    return model
        
class MultilabelDataset(Dataset):
    def __init__(self, file_paths, img_transform):
        super(Dataset).__init__()
        self.file_paths = file_paths 
        self.preprocess = img_transform   
        
    def __getitem__(self, index):
        data = torch.load(self.file_paths[index])
        img, olabel, mlabel = data['image'], data['original_label'], torch.cat((data['correct_multi_labels'], data['unclear_multi_labels']), dim=0)
        
        img = img.to(torch.float32) / 255.0 # unit8 -> float32
        img = self.preprocess(img)

        mlabel = mlabel.to(torch.int64)       # int32 -> int64
        size = mlabel.shape[0]
        if size < 10:
            padding = torch.full((10 - size,), -1, dtype=mlabel.dtype)
            mlabel = torch.cat((mlabel, padding))
        elif size > 10:
            mlabel = mlabel[:10]
        else:
            pass

        # mlabel = torch.squeeze(mlabel)        # [batch_size, 1] -> [batch_size]
        # print(mlabel.shape)
        
        return img, olabel, mlabel
                
    def __len__(self):
        return len(self.file_paths)

if __name__ == "__main__":
    # Args
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--cuda", required=False, type=int, default=0,
                        choices=[0,1,2,3,4,5,6,7], help="Enter a GPU device id from 0 to 7")
    args = parser.parse_args()
    
    # Device
    device = torch.device('cuda:' + str(args.cuda))

    # Data
    output_file = '/media/data_cifs/pfeng2/Pseudo_ClickMe/Results/multi_label_results.csv'
    data_dir = "/media/data_cifs/pfeng2/Harmoization/datasets/imagenet_multi_label"
    ckpt_cache = '/media/data_cifs/pfeng2/timm_model_zoo/'
    file_paths = glob.glob(os.path.join(data_dir, '*.pth')) 
    
    links = {
        # "resnetv2_50.a1h_in1k": "resnetv2_50.a1h_in1k_harmonized",
        # "vit_tiny_patch16_224.augreg_in21k_ft_in1k": "vit_tiny_patch16_224.augreg_in21k_ft_in1k_harmonized",
        # "convnext_tiny.fb_in1k": "convnext_tiny.fb_in1k_harmonized",
        # "mobilenetv3_small_050.lamb_in1k": "mobilenetv3_small_050.lamb_in1k_harmonized",
        "resnet18.tv_in1k": "resnet18.tv_in1k_harmonized",
        "resnet34.tv_in1k": "resnet34.tv_in1k_harmonized",
        "resnet50.tv_in1k": "resnet50.tv_in1k_harmonized",
        "resnet101.tv_in1k": "resnet101.tv_in1k_harmonized",
        "resnet152.tv_in1k": "resnet152.tv_in1k_harmonized",
    }
    # Convert to a single list (key-value flattened)
    model_names = ["mobilenetv3_small_050.lamb_in1k_harmonized"] + [item for pair in links.items() for item in pair]


    for i, model_name in enumerate(model_names):
        # Load models
        

        model = load_model(model_name, device)
        model.eval()

        # Get input configs
        data_config = timm.data.resolve_model_data_config(model)
        img_transform = create_transform(**data_config)
        img_transform = transforms.Compose(
            [transforms.ToPILImage()] + img_transform.transforms)
        # print(img_transform)
        
        # Create dataset
        dataset = MultilabelDataset(file_paths, img_transform)
        dataloader = DataLoader(dataset, batch_size=1, num_workers=1, pin_memory=True)
        start = time.time()
        cnt = 0

        num_correct_per_class, num_images_per_class = {}, {}
        for batch_id, (img, olabel, mlabel) in enumerate(dataloader):
            print("  batch id: %s | %s/%s | %s | CUDA: %s\r" % (batch_id, str(i+1), len(model_names), model_name, str(args.cuda)), end = "")

            img, olabel, mlabel = img.to(device, non_blocking=True), olabel.to(device, non_blocking=True), mlabel.to(device, non_blocking=True)

            # The label of the image in ImageNet
            cur_class = olabel.item()

            # If we haven't processed this class yet, set the counters to 0
            if cur_class not in num_correct_per_class:
                num_correct_per_class[cur_class] = 0
                num_images_per_class[cur_class] = 0

            num_images_per_class[cur_class] += 1

            # Get the predictions for this image
            with torch.no_grad():
                output = model(img)
                # print(output.shape)
                cur_pred = torch.argmax(output, axis=-1) # get the index of the max log-probability)
                # print('\n',cur_pred, olabel)

            # Check prediction
            if torch.any(mlabel == cur_pred.item()):
                num_correct_per_class[cur_class] += 1

        acc_avg = 0
        num_classes = 1000
        assert len(num_correct_per_class) == num_classes
        assert len(num_images_per_class) == num_classes
        for cid in range(num_classes):
            acc_avg += num_correct_per_class[cid] / num_images_per_class[cid]
        acc_avg /= num_classes
   
        end = time.time()
        print("") 

        record = [model_name, round(acc_avg, 4), int(end-start)]
        print(record)

        # Write info
        write_csv_all(record, output_file)
        
        print(model_name, "has been evaluated!\n--------------------------")