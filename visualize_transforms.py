import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
from timm.data import create_loader_scale, resolve_data_config, create_dataset, create_loader
from timm.models import create_model
import torchvision.transforms as transforms
import numpy as np
from PIL import Image

# Constants from train_skeleton_dl.py
CSV_FILE = "/cifs/data/tserre_lrs/projects/projects/prj_concept_surgery/finetuning_models/fp_checked2.csv"
ROOT_DIR = "/oscar/data/tserre/npant1/ILSVRC/train"
MASK_LOOKUP_JSON = '/users/irodri15/data/irodri15/Hmax/pytorch-image-models/timm/data/_info/image_to_mask_lookup.json'

def save_image(image, filename, label=None, scale_band=None):
    """Save a tensor image to file with optional label and scale band information"""
    # Convert the image to numpy array and transpose to HWC format
    image_np = image.cpu().numpy().transpose(1, 2, 0)
    
    # Normalize the image to [0, 255] range
    image_np = (image_np - image_np.min()) / (image_np.max() - image_np.min())
    image_np = (image_np * 255).astype('uint8')
    
    # Create figure with title if label or scale_band is provided
    if label is not None or scale_band is not None:
        plt.figure(figsize=(10, 10))
        plt.imshow(image_np)
        title = []
        if label is not None:
            title.append(f'Label: {label}')
        if scale_band is not None:
            title.append(f'Scale Band: {scale_band}')
        plt.title('\n'.join(title))
        plt.axis('off')
        plt.savefig(filename, bbox_inches='tight', pad_inches=0.1)
        plt.close()
    else:
        plt.imsave(filename, image_np)
    
    print(f"Saved image to {filename}")

def visualize_transforms():
    # Create output directory
    output_dir = "transform_visualizations"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create model to access the transformation functions
    model = create_model('chresmax_v3_2_dl', num_classes=1000, ip_scale_bands=4)
    
    # Extract data_config using resolve_data_config
    data_config = resolve_data_config({'input_size': (3, 227, 227)}, model=model)
    
    # Create training loader with is_training=True
    loader_train = create_loader_scale(
        csv_file=CSV_FILE,
        root_dir=ROOT_DIR,
        mask_look_up_json=MASK_LOOKUP_JSON,
        root=ROOT_DIR,
        input_size=data_config['input_size'],
        batch_size=4,
        is_training=True,
        no_aug=False,
        interpolation=data_config['interpolation'],
        mean=data_config['mean'],
        std=data_config['std'],
        num_workers=4,
        device='cpu',
        use_prefetcher=False
    )
    
    # Create evaluation dataset
    dataset_eval = create_dataset(
        'imagenet',
        root=ROOT_DIR,
        split='validation',
        is_training=False,
        batch_size=4,
        input_img_mode='RGB',
    )
    
    # Create evaluation loader
    loader_eval = create_loader(
        dataset_eval,
        input_size=data_config['input_size'],
        batch_size=4,
        is_training=False,
        interpolation=data_config['interpolation'],
        mean=data_config['mean'],
        std=data_config['std'],
        num_workers=4,
        distributed=False,
        crop_pct=data_config['crop_pct'],
        pin_memory=False,
        device='cpu',
        use_prefetcher=False
    )
    
    try:
        # Get a batch from training loader
        for batch_idx, (input_train, target_train, scale_band_train, center_train) in enumerate(loader_train):
            if batch_idx >= 1:  # Just process first batch
                break
                
            # Print batch information for training
            print("\nTraining Batch Information:")
            print(f"Target labels: {target_train}")
            print(f"Scale bands: {scale_band_train}")
            print(f"Center coordinates: {center_train}")
                
            # Save original training images with labels
            for i in range(input_train.size(0)):
                save_image(
                    input_train[i], 
                    f"{output_dir}/train_original_{i}.png",
                    label=target_train[i].item(),
                    scale_band=scale_band_train[i].item()
                )
            
            # Test make_reference_scale on training batch
            x_reference_train = model.model_backbone.make_reference_scale(input_train, model.ip_scale_bands, scale_band_train)
            for i in range(x_reference_train.size(0)):
                save_image(
                    x_reference_train[i], 
                    f"{output_dir}/train_reference_scale_{i}.png",
                    label=target_train[i].item(),
                    scale_band=scale_band_train[i].item()
                )
            
            # Test make_ip on training batch
            ip_output_train = model.model_backbone.make_ip(input_train, model.ip_scale_bands)
            for i, scale_img in enumerate(ip_output_train):
                for j in range(min(2, scale_img.size(0))):  # Save first 2 images from each scale
                    save_image(
                        scale_img[j], 
                        f"{output_dir}/train_ip_scale_{i}_img_{j}.png",
                        label=target_train[j].item(),
                        scale_band=scale_band_train[j].item()
                    )
        
        # Get a batch from evaluation loader
        for batch_idx, (input_eval, target_eval) in enumerate(loader_eval):
            if batch_idx >= 1:  # Just process first batch
                break
                
            # Print batch information for evaluation
            print("\nEvaluation Batch Information:")
            print(f"Target labels: {target_eval}")
                
            # Save original evaluation images with labels
            for i in range(input_eval.size(0)):
                save_image(
                    input_eval[i], 
                    f"{output_dir}/eval_original_{i}.png",
                    label=target_eval[i].item()
                )
            
            # Test make_reference_scale on evaluation batch
            x_reference_eval = model.model_backbone.make_reference_scale(input_eval, model.ip_scale_bands, torch.zeros(input_eval.size(0), dtype=torch.long))
            for i in range(x_reference_eval.size(0)):
                save_image(
                    x_reference_eval[i], 
                    f"{output_dir}/eval_reference_scale_{i}.png",
                    label=target_eval[i].item()
                )
            
            # Test make_ip on evaluation batch
            ip_output_eval = model.model_backbone.make_ip(input_eval, model.ip_scale_bands)
            for i, scale_img in enumerate(ip_output_eval):
                for j in range(min(2, scale_img.size(0))):  # Save first 2 images from each scale
                    save_image(
                        scale_img[j], 
                        f"{output_dir}/eval_ip_scale_{i}_img_{j}.png",
                        label=target_eval[j].item()
                    )
            
            # Perform tensor comparison between training and evaluation batches
            print("\nTensor Comparison:")
            print(f"Training input shape: {input_train.shape}")
            print(f"Evaluation input shape: {input_eval.shape}")
            print(f"Training and evaluation inputs are identical: {torch.allclose(input_train, input_eval)}")
            print(f"Training and evaluation targets are identical: {torch.allclose(target_train, target_eval)}")
            
            # Print statistics for training inputs
            print("\nTraining Input Statistics:")
            print(f"Min: {input_train.min().item()}")
            print(f"Mean: {input_train.mean().item()}")
            print(f"Max: {input_train.max().item()}")
            print(f"Std: {input_train.std().item()}")
            
            # Print statistics for evaluation inputs
            print("\nEvaluation Input Statistics:")
            print(f"Min: {input_eval.min().item()}")
            print(f"Mean: {input_eval.mean().item()}")
            print(f"Max: {input_eval.max().item()}")
            print(f"Std: {input_eval.std().item()}")
            
            # Print statistics for training targets
            print("\nTraining Target Statistics:")
            print(f"Min: {target_train.min().item()}")
            print(f"Mean: {target_train.float().mean().item()}")
            print(f"Max: {target_train.max().item()}")
            
            # Print statistics for evaluation targets
            print("\nEvaluation Target Statistics:")
            print(f"Min: {target_eval.min().item()}")
            print(f"Mean: {target_eval.float().mean().item()}")
            print(f"Max: {target_eval.max().item()}")
            
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    visualize_transforms() 