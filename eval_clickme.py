import argparse
import os
import torch
import numpy as np
from torchvision import transforms
import timm
from timm.data import resolve_data_config, create_transform

from harmonization.common import load_clickme_val
from harmonization.evaluation import evaluate_clickme

import matplotlib.pyplot as plt 


import sys

sys.path.append("/media/data_cifs/projects/prj_hmax_masks/pytorch-image-models-alexmax/timm/")

from models.RESMAX import resmax_bypass, resmax_v2
from models.ALEXMAX3 import chalexmax_v3_3
from models.ALEXMAX import VGGMAX_V1, vggmax_v1, S1_VGG_Big
from models.ALEXMAX import *
from models.cornet import CORnet_S, cornet_s
from xplique.plots import plot_attributions

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


from torchvision import transforms
from timm.data import resolve_data_config, create_transform




# -- MODEL LOADERS --
def load_resmax_bypass():
    checkpoint_path = "/media/data_cifs/prj_hmax/models/debug_resmax_bypass_concat_gpu_8_cl_0_ip_3_322_322_18432_c1[_6,3,1_]/model_best.pth.tar"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = resmax_bypass(num_classes=1000, classifier_input_size=18432)
    model.load_state_dict(checkpoint['state_dict'])
    return model

def load_chalexmax_v3_3():
    ckpt_path = "/media/data_cifs/prj_hmax/models/debug5_resize2_{chalexmax_v3_3}_cl_{1}_ip_{7}_322_{9216}/model_best.pth.tar"
    model = chalexmax_v3_3(num_classes=1000, in_chans=3, classifier_input_size=9216,ip_scale_bands=7,contrastive_loss=True, cl_lambda=0.1).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["state_dict"], strict=False)
    model.to(device).eval()
    return model


def load_cornet():
    checkpoint_path = "/media/data_cifs/prj_hmax/models/cornet_s_debug/model_best.pth.tar"
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = cornet_s().to(device)
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    return model
    
def load_VGGMAX():
    checkpoint_path = "/media/data_cifs/prj_hmax/models/ip_3_vggmax_v1_gpu_8_cl_0_ip_3_322_322_20736_c1[_6,3,1_]_scale_0.08/model_best.pth.tar"
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = vggmax_v1(num_classes=1000, big_size=322, small_size=227, in_chans=3, 
                 ip_scale_bands=3, classifier_input_size=20736, contrastive_loss=False, pyramid=False       
    ).to(device).eval()
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    return model 

def load_resmax_v2():
    
    checkpoint_path = "/media/data_cifs/projects/prj_concept_surgery/finetuning_models/Imagenet_harmonization/models/ip_3_vggmax_v1_gpu_8_cl_0_ip_3_322_322_20736_c1[_6,3,1_]_scale_0.08/model_best.pth.tar"
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = resmax_v2(num_classes=1000, big_size=322, small_size=227, in_chans=3, 
                 ip_scale_bands=1, classifier_input_size=13312, contrastive_loss=False, pyramid=False,
                 bypass=False, main_route=False,
                 c_scoring='v2'      
    ).to(device).eval()
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    return model 

def load_resmax_v2_bypass():
    checkpoint_path = "/media/data_cifs/prj_hmax/models/ip_3_resmax_v2_gpu_8_cl_0_ip_3_322_322_18432_c1[_6,3,1_]_bypass/model_best.pth.tar"   
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = resmax_v2(num_classes=1000, big_size=322, small_size=227, in_chans=3, 
                 ip_scale_bands=3, classifier_input_size=18432, contrastive_loss=False, pyramid=False,
                 bypass=True, main_route=False,
                 c_scoring='v2'      
    ).to(device).eval()
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    return model
    
def get_model(model_name):
    if model_name == "CHALEXMAX":
        return load_chalexmax_v3_3()
    elif model_name == "RESMAX_bypass":
        return load_resmax_bypass()
    elif model_name == "RESNET50":
        return torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True).to(device)
    elif model_name == "CORNET":
        return load_cornet() 
    elif model_name == "VGGMAX":
        return load_VGGMAX()
    elif model_name == "RESMAX_V2_BYPASS":
        return load_resmax_v2_bypass()
    
    #elif model_name=="RESNET18":
        #return model = timm.create_model('resnet18.a1_in1k', pretrained=True) # TODO: Input_size_check 
    elif model_name == "ALEXNET":
        return torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', pretrained=True).to(device)
    elif model_name =="VGG":
        return timm.create_model('vgg16.tv_in1k', pretrained=True).to(device)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

# scale up saliency 
def torch_explainer(xbatch, ybatch):
    xbatch = torch.stack([model_preprocess(x) for x in xbatch.numpy().astype(np.uint8)])
    ybatch = torch.tensor(ybatch.numpy())

    xbatch = xbatch.to(device).requires_grad_()
    ybatch = ybatch.to(device)

    if ybatch.ndim > 1:
        ybatch = torch.argmax(ybatch, dim=1)

    model.zero_grad()
    out = model(xbatch)
    logits = out[0] if isinstance(out, (list, tuple)) else out

    output = logits[range(len(ybatch)), ybatch].sum()
    output.backward()

    saliency, _ = torch.max(xbatch.grad.data.abs(), dim=1)
    
    
    return saliency.detach().cpu().numpy()



import tensorflow as tf

TARGET_SIZE = 322

def resize_clickme_batch(images, heatmaps, labels):
    images = tf.image.resize(images, (TARGET_SIZE, TARGET_SIZE))
    heatmaps = tf.image.resize(heatmaps, (TARGET_SIZE, TARGET_SIZE))
    return images, heatmaps, labels

def prepare_clickme_dataset():
    dataset = load_clickme_val(batch_size=128)
    return dataset.map(resize_clickme_batch, num_parallel_calls=tf.data.AUTOTUNE)

# -- MAIN --
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
        
"""
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
