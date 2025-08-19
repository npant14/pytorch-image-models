import argparse
import os
import torch

import sys
sys.path.append("/users/xyu110/pytorch-image-models")
import timm
from timm.models.RESMAX import chresmax_v3_2_abs, chresmax_v3_2
from timm.models.alexnet import alexnet
from timm.models.resnet import resnet18

# Import our Pasupathy class
from pasupathy import Pasupathy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_all_layer_names(model):
    """Extract all layer names from a model"""
    layer_names = []
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf modules only
            layer_names.append(name)
    return layer_names

def load_chresmax_v3_2_abs():
    checkpoint_path = '/oscar/data/tserre/xyu110/pytorch-output/train/0/final_versions/ip_3_chresmax_v3_2_abs_gpu_8_cl_0.1_ip_3_322_322_18432_c1[_6,3,1_]_bypass/model_best.pth.tar'
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = chresmax_v3_2_abs(num_classes=1000, big_size=322, small_size=322, in_chans=3, 
                 ip_scale_bands=3, classifier_input_size=18432, pyramid=False,
                 bypass=True, main_route=False,validation=True,
                 c_scoring='v2'      
    ).to(device).eval()
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    return model

def load_chresmax_v3_2():
    checkpoint_path = '/oscar/data/tserre/xyu110/pytorch-output/train/0/final_versions/ip_3_chresmax_v3_2_gpu_8_cl_0.1_ip_3_322_322_18432_c1[_6,3,1_]_bypass/model_best.pth.tar'
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = chresmax_v3_2(num_classes=1000, big_size=322, small_size=322, in_chans=3, 
                 ip_scale_bands=3, classifier_input_size=18432, pyramid=False,
                 bypass=True, main_route=False,validation=True,
                 c_scoring='v2'      
    ).to(device).eval()
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    return model

# -- MODEL LOADERS --
def load_alexnet_with_aug():   
    checkpoint_path = "/oscar/data/tserre/xyu110/pytorch-output/train/0/baseline_w_aug/ip_0_alexnet_gpu_2_cl_0_ip_3_227_227_0_c1[_6,3,1_]_scale_0.08/model_best.pth.tar"
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = alexnet(channel_size=227).to(device).eval()
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    return model

def load_alexnet_without_aug():   
    checkpoint_path = "/oscar/data/tserre/xyu110/pytorch-output/train/0/baseline_wo_aug/ip_0_alexnet_gpu_2_cl_0_ip_3_227_227_0_c1[_6,3,1_]/model_best.pth.tar"
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = alexnet(channel_size=227).to(device).eval()
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    return model

def load_resnet_without_aug():   
    checkpoint_path = "/oscar/data/tserre/xyu110/pytorch-output/train/0/baseline_wo_aug/ip_0_resnet18_gpu_8_cl_0_ip_3_227_227_512_c1[_6,3,1_]/model_best.pth.tar"
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = resnet18(channel_size=227).to(device).eval()
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    return model

def load_resnet_with_aug():   
    checkpoint_path = "/oscar/data/tserre/xyu110/pytorch-output/train/0/baseline_w_aug/ip_0_resnet18_gpu_8_cl_0_ip_3_227_227_512_c1[_6,3,1_]_scale_0.08/model_best.pth.tar"
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = resnet18(channel_size=227).to(device).eval()
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    return model

    
def get_model(model_name):
    if model_name == "RESNET50":
        return torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True).to(device)
    elif model_name == 'ALEXNET-NO_AUG':
        return load_alexnet_without_aug()
    elif model_name == "ALEXNET-AUG":
        return load_alexnet_with_aug()
    elif model_name == "RESNET18-NO_AUG":
        return load_resnet_without_aug()
    elif model_name == "RESNET18-AUG":
        return load_resnet_with_aug()
    elif model_name == "CHRESMAX_V3_2_ABS":
        return load_chresmax_v3_2_abs()
    elif model_name == "CHRESMAX_V3_2":
        return load_chresmax_v3_2()
    else:
        raise ValueError(f"Unsupported model: {model_name}")

# Define the models to evaluate
hmax_models = [
    "RESNET50",
    "ALEXNET-NO_AUG",
    "ALEXNET-AUG",
    "RESNET18-NO_AUG",
    "RESNET18-AUG",
    "CHRESMAX_V3_2_ABS",
    "CHRESMAX_V3_2"
]

baselines = [
    "RESNET50",
    "ALEXNET-NO_AUG",
    "ALEXNET-AUG",
    "RESNET18-NO_AUG",
    "RESNET18-AUG",
]

# Pasupathy data directory
PASUPATHY_DATA_DIR = "/users/xyu110/scratch/subplots"
OUTPUT_DIR = "./pasupathy_results"

def evaluate_model_on_pasupathy(model_name, layer_name=None):
    """Evaluate a single model on Pasupathy experiment"""
    try:
        print(f"Loading model: {model_name}")
        print(f"Calling get_model({model_name})...")
        model = get_model(model_name).to(device).eval()
        print(f"Model {model_name} loaded successfully!")
        
        imgsize = 322
        if model_name in baselines:
            imgsize = 227
        
        # Get all available layers if no specific layer is provided
        if layer_name is None:
            layer_names = get_all_layer_names(model)
        else:
            layer_names = [layer_name]
        
        print(f"Will evaluate {len(layer_names)} layers for {model_name}")
        
        results = []
        
        # Set up results file path
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        results_file = os.path.join(OUTPUT_DIR, "all_pasupathy_scores.csv")
        
        for layer in layer_names:
            try:
                print(f"Evaluating {model_name} on layer: {layer}")

                # Set up Pasupathy experiment
                pasupathy_exp = Pasupathy(
                    model=model, 
                    outdir=OUTPUT_DIR, 
                    device=device, 
                    data_dir=PASUPATHY_DATA_DIR, 
                    img_size=imgsize, 
                    layer=layer
                )
                
                # Run the experiment
                pasupathy_score = pasupathy_exp.run()
                
                print(f"Pasupathy Score for {model_name}, {layer}: {pasupathy_score}")
                results.append((model_name, layer, pasupathy_score))
                
                # Save result to file
                with open(results_file, "a") as f:
                    f.write(f"{model_name},{layer},{pasupathy_score:.4f}\n")
                
                # Clean up GPU memory
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                print(f"Error evaluating {model_name} on layer {layer}: {e}")
                results.append((model_name, layer, "FAILED"))
                with open(results_file, "a") as f:
                    f.write(f"{model_name},{layer},FAILED\n")
                continue
        
        return results
        
    except Exception as e:
        print(f"Error loading model {model_name}: {e}")
        return [(model_name, "N/A", "FAILED")]

def evaluate_all_models():
    """Evaluate all HMAX models on Pasupathy experiment"""
    print("Starting Pasupathy evaluation for all HMAX models on all layers...")
    
    # Initialize CSV file with header
    results_file = os.path.join(OUTPUT_DIR, "all_pasupathy_scores.csv")
    with open(results_file, "w") as f:
        f.write("model,layer,score\n")
    
    all_results = []
    
    for model_name in hmax_models:
        print(f"\n{'='*50}")
        print(f"Evaluating: {model_name}")
        print(f"{'='*50}")
        
        model_results = evaluate_model_on_pasupathy(model_name)
        all_results.extend(model_results)
        
        # Clean up GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Print summary
    print(f"\n{'='*50}")
    print("SUMMARY OF RESULTS")
    print(f"{'='*50}")
    print("model,layer,score")
    for model_name, layer, score in all_results:
        if isinstance(score, float):
            print(f"{model_name},{layer},{score:.4f}")
        else:
            print(f"{model_name},{layer},{score}")
    
    return all_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate HMAX models on Pasupathy experiment")
    parser.add_argument('--model', type=str, help="Specific model to evaluate (optional)")
    parser.add_argument('--layer', type=str, help="Specific layer to evaluate (optional)")
    args = parser.parse_args()
    
    if args.model:
        # Evaluate specific model
        if args.model not in hmax_models:
            print(f"Error: {args.model} not in available models: {hmax_models}")
            exit(1)
        
        print(f"Evaluating specific model: {args.model}")
        results = evaluate_model_on_pasupathy(args.model, args.layer)
        
        print("\nResults:")
        print("model,layer,score")
        for model_name, layer, score in results:
            if isinstance(score, float):
                print(f"{model_name},{layer},{score:.4f}")
            else:
                print(f"{model_name},{layer},{score}")
    else:
        # Evaluate all models
        evaluate_all_models() 