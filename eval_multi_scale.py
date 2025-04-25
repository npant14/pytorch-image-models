import argparse
import os
import torch
import numpy as np
from torchvision import transforms
import timm
from timm.data import resolve_data_config, create_transform
from harmonization.common import load_clickme_val
from harmonization.evaluation import evaluate_clickme
from clickme_scaling import * 
import matplotlib.pyplot as plt 
from pad import CenterResizeCropPad
import torchvision
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import tensorflow as tf
import sys
from loading_models import *
from xplique.plots import plot_attributions
from torchvision.transforms import InterpolationMode
from xplique.wrappers import TorchWrapper
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from torchvision import transforms
from timm.data import resolve_data_config, create_transform
import torch.nn.functional as F
import torchvision.transforms as T
import numpy as np

from models.HMAX import get_ip_scales, pad_to_size
sys.path.append("/cifs/data/tserre_lrs/projects/projects/prj_hmax_masks/pytorch-image-models-alexmax/timm/")
                

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])


def preprocess_input(images):
    """
    Preprocesses images for the harmonized models.
    The images are expected to be in RGB format with values in the range [0, 255].

    Parameters
    ----------
    images
        Tensor or numpy array to be preprocessed.
        Expected shape (N, W, H, C).

    Returns
    -------
    preprocessed_images
        Images preprocessed for the harmonized models.
    """
    images = images / 255.0

    images = images - IMAGENET_MEAN
    images = images / IMAGENET_STD

    return images

def pad_or_crop(img_tensor, target_size=322):
    """
    Pads or crops a 3D (C, H, W) or 4D (B, C, H, W) image tensor to the target size.
    
    Parameters:
        img_tensor (Tensor): Input tensor of shape (C, H, W) or (B, C, H, W).
        target_size (int): Desired height and width after padding or cropping.
        
    Returns:
        Tensor: Tensor padded or cropped to (target_size, target_size).
    """
    is_batched = img_tensor.dim() == 4
    if not is_batched:
        img_tensor = img_tensor.unsqueeze(0)  # add batch dimension

    _, _, h, w = img_tensor.shape
    
    img_hw = img_tensor.shape[-1]
    new_hw = int(target_size)
    x_rescaled = F.interpolate(img_tensor, size=(new_hw, new_hw), mode='nearest')

    if new_hw <= img_hw:
        # pad if smaller
        x_rescaled = pad_to_size(x_rescaled, (img_hw, img_hw))
    else:
        # center-crop if bigger
        center_crop = torchvision.transforms.CenterCrop(img_hw)
        x_rescaled = center_crop(x_rescaled)

    return x_rescaled 


def pad_or_crop_tf(tensor, target_size=322):
    """
    Pad or crop a single 3D tensor (H, W, C) to target size.
    Assumes `tensor` is a 3D float32 tensor.
    """
    shape = tf.shape(tensor)
    height, width = shape[0], shape[1]
    
    
    # Pad if smaller
    if height<target_size and width < target_size:
        pad_height = tf.maximum(target_size - height, 0)
        pad_width = tf.maximum(target_size - width, 0)
        tensor = tf.pad(tensor, [[pad_height // 2, pad_height - pad_height // 2],
                                [pad_width // 2, pad_width - pad_width // 2],
                                [0, 0]])

    else: 
        tensor = tf.image.crop(tensor, target_size, target_size) # f.

    return tensor

DIR_ROOT= "/cifs/data/tserre_lrs/projects/prj_hmax/models/"
#DIR_ROOT = "/files22_lrsresearch/CLPS_Serre_Lab/prj_hmax/models/" # CCV

#DIR_ROOT = "/media/data_cifs/prj_hmax/models/" # - Z node
# scale up saliency 
def torch_explainer(xbatch, ybatch):
    
    # Preprocess batch
    xbatch = torch.stack([model_preprocess(x) for x in xbatch.numpy().astype(np.uint8)])
    
    xbatch = xbatch.permute(0, 1, 2, 3)

    # Resize using the provided scale
    xbatch = pad_or_crop(xbatch, target_size=322)   

    ybatch = torch.tensor(ybatch.numpy())
    xbatch = xbatch.to(device).detach().clone().requires_grad_(True)
    ybatch = ybatch.to(device)

    if ybatch.ndim > 1:
        ybatch = torch.argmax(ybatch, dim=1)

    model.zero_grad()
    out = model(xbatch)
    logits = out[0] if isinstance(out, (list, tuple)) else out

    output = logits[range(len(ybatch)), ybatch].sum()
    #print("Logits:", logits[range(len(ybatch)), ybatch])
    output.backward()

    assert xbatch.grad is not None, "Gradient not populated!"
    #print("Gradients:", xbatch.grad.abs().sum())

    saliency, _ = torch.max(xbatch.grad.data.abs(), dim=1)

    #print(f"[torch_explainer] After backward, saliency shape: {saliency.shape}")  # (B, 322, 322)
    return saliency.detach().cpu().numpy()



from xplique.attributions import Saliency

import tensorflow as tf

TARGET_SIZE = 322
# 322

SCALES= [160, 192, 227, 270, 322, 382, 454]
scale_factor_list = [0.49, 0.59, 0.707, 0.841, 1.0, 1.189, 1.414, 1.681, 2.0]

def resize_clickme_batch(images, heatmaps, labels, scale):
    
    
    # Resize to intermediate scale
    images = tf.image.resize(images, (scale, scale), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR )
    heatmaps = tf.image.resize(heatmaps, (scale, scale), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    # Then pad/crop to 322x322
    images = tf.map_fn(lambda img: pad_or_crop_tf(img, TARGET_SIZE), images)
    heatmaps = tf.map_fn(lambda hm: pad_or_crop_tf(hm, TARGET_SIZE), heatmaps)

    return images, heatmaps, labels


def prepare_clickme_dataset(TARGET_SIZE):
    dataset = load_clickme_val(batch_size=64)
    
    return dataset.map(
        lambda x, y, z, TARGET_SIZE: resize_clickme_batch(x, y, z, TARGET_SIZE),
        num_parallel_calls=tf.data.AUTOTUNE
    )

def visualize_saliency_across_scales(model, model_name, dataset_dict, scales):
    fig, axs = plt.subplots(4, len(scales), figsize=(len(scales) * 3, 10))
    plt.subplots_adjust(wspace=0.1, hspace=0.3)

    for col, scale in enumerate(scales):
        dataset = dataset_dict[scale]
        image_tf, heatmap_tf, label_tf = next(iter(dataset.take(1)))

        image_np = image_tf.numpy()[0]  # shape (H, W, C)
        label_np = label_tf.numpy()[0]

        # Saliency
        saliency_map = torch_explainer(image_tf, label_tf)[0]  # (H, W)

        # Normalize saliency for overlay
        saliency_norm = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min() + 1e-5)

        axs[0, col].imshow(image_np.astype(np.uint8))
        axs[0, col].set_title(f"Scale {scale}")
        axs[1, col].imshow(heatmap_tf.numpy()[0], cmap="viridis")
        axs[2, col].imshow(saliency_map, cmap="hot")
        axs[3, col].imshow(image_np.astype(np.uint8))
        axs[3, col].imshow(saliency_norm, cmap='hot', alpha=0.4)

        for row in range(4):
            axs[row, col].axis('off')

    save_dir = f"/files22_lrsresearch/CLPS_Serre_Lab/projects/prj_concept_surgery/finetuning_models/debugvis_grid/"
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f"{model_name}_saliency_grid.png"), bbox_inches='tight')
    plt.close()

# -- MAIN --

from clickme_scaling import prepare_clickme_dataset_at_scale
SCALES = [160, 192, 227, 270, 322, 382, 454]
BATCH_SIZE = 128


#dataset_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate model on ClickMe across scales.")
    parser.add_argument('--model', type=str, required=True,
                        choices=["CHALEXMAX", "RESMAX_bypass", "RESNET50", "CORNET", "VGGMAX", "RESMAX_V2_BYPASS", "VGG", "ALEXNET-NO_AUG", "ALEXNET-AUG" ])
    args = parser.parse_args()

    model = get_model(args.model).to(device).eval()
    print(f"Loaded model: {args.model}")
    
    wrapped_model = TorchWrapper(model, device)

    # Evaluate on each scale
    results_file = "/files22_lrsresearch/CLPS_Serre_Lab/projects/prj_concept_surgery/finetuning_models/model_alignment_scores_by_scale_pp.txt"
    
    scale_to_factor = dict(zip(SCALES, scale_factor_list))

    dataset_dict = {} 

        
    with open(results_file, "a") as f:
                
        for TARGET_SIZE in SCALES:
            
            scale = TARGET_SIZE
            
            batch_size = 64
            #print(f"\n--- Evaluating at scale: {scale} ---")
            dataset = prepare_clickme_dataset(scale)
            dataset_dict[scale] = dataset
            
            config = resolve_data_config({'input_size': (3, scale, scale)}, model=model)
            transform = create_transform(**config)
            

            IMAGENET_MEAN = [0.485, 0.456, 0.406]
            IMAGENET_STD = [0.229, 0.224, 0.225]            
            
            
            model_preprocess =  transforms.Compose([transforms.ToPILImage(),
                                                    transforms.Resize((scale, scale), interpolation=InterpolationMode.NEAREST),
                                                    transforms.ToTensor(),  # Converts [0, 255] to [0.0, 1.0]
                                                    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)  # Normalizes using ImageNet stats
                                                    create_transform(**config)])            
            
     
            # TODO: Debug why 0 values in case of scale = 454 leading to nan 
            
            print(f"\n--- Evaluating harmonization at scale: {scale} ---")
            
            scores = evaluate_clickme(
                model=model,
                explainer=lambda x, y: torch_explainer(x, y),
                clickme_val_dataset=dataset                
            )        
            
            alignment_score = scores['alignment_score']
            print(f"Scale {scale} | Alignment Score: {alignment_score:.4f}")
            f.write(f"{args.model} | Scale {scale} | Alignment: {alignment_score:.4f}\n")
            
            
#visualize_saliency_across_scales(model, args.model, dataset_dict, SCALES)
       
                        
                
"""             
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate model on ClickMe.")
    parser.add_argument('--model', type=str, required=True, choices=["CHALEXMAX", "RESMAX", "RESNET", "ALEXNET", "CORNET", "VGGMAX", "RESMAX_V2_BYPASS"],
                    help="Model to evaluate")
    args = parser.parse_args()

    #model = timm.create_model('resnet18.a1_in1k', pretrained=True)
    #model = model.to(device).eval()    
    
    
    model= get_model(args.model).to(device).eval()
    print("Model loaded")
    
    

    # Set up preprocessing
    config = resolve_data_config({'input_size': (3, 322, 322)}, model=model)
    transform = create_transform(**config)
    
    model_preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((322, 322)), # Nearest neighbor 
        transform
    ])
    

    # Load and resize dataset
    clickme_dataset = prepare_clickme_dataset()
    
    print("Model evaluation beginning")

    # Evaluate
    scores = evaluate_clickme(model,
                              explainer=torch_explainer,
                              clickme_val_dataset=clickme_dataset)
    
    print("print model evaluation ended")
    
        
        
    
    print("Alignment Score:", scores['alignment_score']) 
    # Append result to file
    results_file = "/media/data_cifs/projects/prj_concept_surgery/finetuning_models/model_alignment_scores.txt"
    with open(results_file, "a") as f:
        f.write(f"{args.model}: {scores['alignment_score']:.4f}\n") 
        

# plot saliency maps 


    for idx, (image_tf, _, label_tf) in enumerate(clickme_dataset_resized.take(10)):
        image_np = image_tf.numpy()
        label_np = label_tf.numpy()

        image_tensor = torch.tensor(image_np).permute(2, 0, 1)
        label_tensor = torch.tensor(label_np)

        saliency = torch_explainer(torch.unsqueeze(image_tensor, 0), torch.unsqueeze(label_tensor, 0))
        plot_attributions(saliency_map=saliency[0], image=image_tensor)
        outdir="./outputs"
        os.makedirs(outdir, exist_ok=True)
        save_path = os.path.join(outdir, f"{idx}_Scaled_img.png")
        plt.savefig(save_path)
"""                
