"""
ImageNet Multi-Label Inference for HMAX Models

This script evaluates HMAX models on the ImageNet multi-label dataset.
It handles custom HMAX model loading and preprocessing requirements.
"""

import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset

import sys
import os

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from timm.models import create_model
from utils_hmax import (
    CenterResizeCropPad,
    load_vit_base,
    load_alexnet_timm,
    load_resnet18_timm,
)

import csv
import argparse
import glob
import time
import warnings
warnings.filterwarnings("ignore")


def write_csv_all(record, path):
    """Write evaluation results to CSV file."""
    header = ['model', 'scale', 'multi_label_acc', 'time']
    file_exists = os.path.isfile(path)

    with open(path, mode='a+', newline='') as csv_file:
        writer = csv.writer(csv_file)
        if not file_exists:
            writer.writerow(header)
        writer.writerow(record)


def load_model(model_name, checkpoint_path, device):
    """
    Load model from checkpoint.
    
    Args:
        model_name: Name of the model ('hmax_v3_adj', 'alexnet_aug', 'resnet18_aug', 'alexnet_wo_aug', 'resnet18_wo_aug')
        checkpoint_path: Path to the model checkpoint
        device: Device to load the model on
    
    Returns:
        Loaded model in eval mode
    """
    print(f"Loading {model_name}" + (f" from {checkpoint_path}" if checkpoint_path else " (pretrained weights)"))
    
    if model_name == 'vit_base':
        # ViT-Base uses pretrained ImageNet weights — no checkpoint file needed
        model = load_vit_base(device=device)
        model.eval()
        return model

    if model_name == 'alexnet_timm':
        # timm AlexNet pretrained on ImageNet-1k
        model = load_alexnet_timm(device=device)
        model.eval()
        return model

    if model_name == 'resnet18_timm':
        # torchvision ResNet-18 weights through timm registry
        model = load_resnet18_timm(device=device)
        model.eval()
        return model

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    if model_name == 'hmax_v3_adj':
        from timm.models.RESMAX import hmax_v3_adj
        model = hmax_v3_adj(
            num_classes=1000,
            big_size=322,
            small_size=322,
            in_chans=3,
            ip_scale_bands=3,
            classifier_input_size=18432,
            pyramid=False,
            bypass=True,
            main_route=False,
            validation=True
        )
    
    elif model_name == 'alexnet_aug':
        from timm.models.alexnet import alexnet
        model = alexnet(channel_size=227)
    
    elif model_name == 'resnet18_aug':
        from timm.models.resnet import resnet18
        model = resnet18(channel_size=227)
    
    elif model_name == 'alexnet_wo_aug':
        from timm.models.alexnet import alexnet
        model = alexnet(channel_size=227)
    
    elif model_name == 'resnet18_wo_aug':
        from timm.models.resnet import resnet18
        model = resnet18(channel_size=227)
    
    else:
        raise ValueError(f"Unknown model name: {model_name}. Choose from: hmax_v3_adj, alexnet_aug, resnet18_aug, alexnet_wo_aug, resnet18_wo_aug")
    
    # Load weights
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    model = model.to(device)
    model.eval()
    
    print(f"Model {model_name} loaded successfully")
    return model


class MultilabelDataset(Dataset):
    """Dataset for ImageNet multi-label evaluation."""
    
    def __init__(self, file_paths, img_transform):
        super(Dataset).__init__()
        self.file_paths = file_paths 
        self.preprocess = img_transform   
        
    def __getitem__(self, index):
        data = torch.load(self.file_paths[index])
        img, olabel, mlabel = data['image'], data['original_label'], torch.cat(
            (data['correct_multi_labels'], data['unclear_multi_labels']), dim=0
        )
        
        # Convert uint8 to float32 and normalize to [0, 1]
        img = img.to(torch.float32) / 255.0
        
        # Apply preprocessing
        img = self.preprocess(img)

        # Convert multi-labels to int64
        mlabel = mlabel.to(torch.int64)
        
        # Pad or truncate to fixed size (10)
        size = mlabel.shape[0]
        if size < 10:
            padding = torch.full((10 - size,), -1, dtype=mlabel.dtype)
            mlabel = torch.cat((mlabel, padding))
        elif size > 10:
            mlabel = mlabel[:10]
        
        return img, olabel, mlabel
                
    def __len__(self):
        return len(self.file_paths)


def get_hmax_transforms(input_size=322, use_multiscale=False, scale=None):
    """
    Get transforms for HMAX models.
    
    Args:
        input_size: Input image size (default: 322 for most HMAX models)
        use_multiscale: Whether to use CenterResizeCropPad for multiscale testing
        scale: Scale to use with CenterResizeCropPad (required if use_multiscale=True)
    
    Returns:
        Transform composition
    """
    if use_multiscale:
        if scale is None:
            raise ValueError("scale must be provided when use_multiscale=True")
        # Use CenterResizeCropPad for multiscale testing
        img_transform = transforms.Compose([
            CenterResizeCropPad(output_size=(input_size, input_size), scale=scale, mode='constant'),
            transforms.Resize((input_size, input_size)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        # Standard transform
        img_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    return img_transform


def evaluate_model(model, dataloader, device, model_name, scale_info=""):
    """
    Evaluate model on multi-label dataset.
    
    Args:
        model: Model to evaluate
        dataloader: DataLoader for the dataset
        device: Device to run inference on
        model_name: Name of the model (for logging)
        scale_info: Additional scale information for logging
    
    Returns:
        Average accuracy across all classes
    """
    num_correct_per_class = {}
    num_images_per_class = {}
    
    model.eval()
    
    for batch_id, (img, olabel, mlabel) in enumerate(dataloader):
        print(f"  Processing batch {batch_id+1}/{len(dataloader)} | {model_name}{scale_info}\r", end="")
        
        img = img.to(device, non_blocking=True)
        olabel = olabel.to(device, non_blocking=True)
        mlabel = mlabel.to(device, non_blocking=True)
        
        # The label of the image in ImageNet
        cur_class = olabel.item()
        
        # Initialize counters for this class if not seen before
        if cur_class not in num_correct_per_class:
            num_correct_per_class[cur_class] = 0
            num_images_per_class[cur_class] = 0
        
        num_images_per_class[cur_class] += 1
        
        # Get predictions
        with torch.no_grad():
            output = model(img)
            
            # Handle tuple output (HMAX models return tuple)
            if isinstance(output, tuple):
                output = output[0]  # Get the first element (main predictions)
            
            cur_pred = torch.argmax(output, dim=-1)
        
        
        # Check if prediction matches any of the valid labels
        if torch.any(mlabel == cur_pred.item()):
            num_correct_per_class[cur_class] += 1
    
    # Calculate average accuracy across all classes
    acc_avg = 0
    num_classes = 1000
    
    # Ensure we have data for all classes
    if len(num_correct_per_class) != num_classes:
        print(f"\nWarning: Only {len(num_correct_per_class)} classes found, expected {num_classes}")
    
    for cid in num_correct_per_class.keys():
        acc_avg += num_correct_per_class[cid] / num_images_per_class[cid]
    
    acc_avg /= len(num_correct_per_class)
    
    return acc_avg


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Evaluate models on ImageNet multi-label dataset')
    parser.add_argument('-m', '--model', type=str, required=True,
                        choices=[
                            'hmax_v3_adj',
                            'alexnet_aug',
                            'resnet18_aug',
                            'alexnet_wo_aug',
                            'resnet18_wo_aug',
                            'alexnet_timm',
                            'resnet18_timm',
                            'vit_base',
                        ],
                        help='Model type to evaluate')
    parser.add_argument('-c', '--checkpoint', type=str, default='',
                        help='Path to model checkpoint (not required for vit_base/alexnet_timm/resnet18_timm)')
    parser.add_argument('--cuda', type=int, default=0, choices=[0,1,2,3,4,5,6,7],
                        help='GPU device id (default: 0)')
    parser.add_argument('--data-dir', type=str, 
                        default='../scratch/imagenet_multi_label',
                        help='Path to ImageNet multi-label dataset')
    parser.add_argument('--output-file', type=str,
                        default='./hmax_multi_label_results.csv',
                        help='Path to output CSV file')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Batch size for evaluation (default: 1)')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers (default: 4)')
    parser.add_argument('--input-size', type=int, default=322,
                        help='Input image size (default: 322)')
    parser.add_argument('--multiscale', action='store_true',
                        help='Run multiscale evaluation with scales [160, 192, 227, 270, 322, 382, 454]')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device(f'cuda:{args.cuda}')
    print(f"Using device: {device}")
    
    # Load data
    print(f"Loading data from {args.data_dir}")
    file_paths = glob.glob(os.path.join(args.data_dir, '*.pth'))
    print(f"Found {len(file_paths)} samples")
    
    if len(file_paths) == 0:
        raise ValueError(f"No .pth files found in {args.data_dir}")
    
    # Load model
    model = load_model(args.model, args.checkpoint, device)
    
    # Define scales for multiscale testing
    scales = [160, 192, 227, 270, 322, 382, 454] if args.multiscale else [args.input_size]
    
    print(f"\nEvaluating {args.model} on {len(file_paths)} images...")
    if args.multiscale:
        print(f"Running multiscale evaluation with scales: {scales}")
    else:
        print(f"Running single scale evaluation with size: {args.input_size}")
    
    # Run evaluation for each scale
    for scale in scales:
        print(f"\n{'='*60}")
        print(f"Testing scale: {scale}")
        print(f"{'='*60}")
        
        # Get transforms for current scale
        if args.multiscale:
            img_transform = get_hmax_transforms(
                input_size=args.input_size, 
                use_multiscale=True, 
                scale=scale
            )
        else:
            img_transform = get_hmax_transforms(
                input_size=args.input_size, 
                use_multiscale=False
            )
        
        # Create dataset and dataloader
        dataset = MultilabelDataset(file_paths, img_transform)
        dataloader = DataLoader(
            dataset, 
            batch_size=args.batch_size, 
            num_workers=args.num_workers, 
            pin_memory=True
        )
        
        # Run evaluation
        start_time = time.time()
        scale_info = f" | Scale: {scale}" if args.multiscale else ""
        acc_avg = evaluate_model(model, dataloader, device, args.model, scale_info)
        end_time = time.time()
        
        elapsed_time = int(end_time - start_time)
        
        print(f"\n\nResults for scale {scale}:")
        print(f"  Model: {args.model}")
        print(f"  Scale: {scale}")
        print(f"  Multi-label Accuracy: {acc_avg:.4f}")
        print(f"  Time: {elapsed_time} seconds")
        
        # Write results to CSV
        record = [args.model, scale, round(acc_avg, 4), elapsed_time]
        write_csv_all(record, args.output_file)
        print(f"  Results written to {args.output_file}")
    
    print(f"\n{'='*60}")
    print("Evaluation complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

# python imagenet_multilabel/multi_label_eval/mla_hmax_inference.py \
#     --model hmax_v3_adj \
#     --checkpoint /path/to/hmax_checkpoint.pth.tar
