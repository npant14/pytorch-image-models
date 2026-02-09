import argparse
import os
import torch
import numpy as np

import sys
sys.path.append("/users/xyu110/pytorch-image-models")
import timm

# Import our Pasupathy class and model loaders
from pasupathy_new import Pasupathy
from rf_analyzer import RFAnalyzer
from utils_hmax import (
    load_chresmax_v3_2_abs,
    load_chresmax_v3_2,
    load_alexnet_with_aug,
    load_alexnet_without_aug,
    load_resnet_without_aug,
    load_resnet_with_aug,
    load_hmax_v3_adj
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_all_layer_names(model):
    """Extract all layer names from a model"""
    layer_names = []
    for name, module in model.named_modules():
        layer_names.append(name)
    return layer_names

    
def get_model(model_name):
    if model_name == "RESNET50":
        return torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True).to(device)
    elif model_name == 'ALEXNET-NO_AUG':
        return load_alexnet_without_aug(device)
    elif model_name == "ALEXNET-AUG":
        return load_alexnet_with_aug(device)
    elif model_name == "RESNET18-NO_AUG":
        return load_resnet_without_aug(device)
    elif model_name == "RESNET18-AUG":
        return load_resnet_with_aug(device)
    elif model_name == "CHRESMAX_V3_2_ABS":
        return load_chresmax_v3_2_abs(device)
    elif model_name == "CHRESMAX_V3_2":
        return load_chresmax_v3_2(device)
    elif model_name == "HMAX_V3_ADJ":
        return load_hmax_v3_adj(device)
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
    "CHRESMAX_V3_2",
    "HMAX_V3_ADJ"
]

baselines = [
    "RESNET50",
    "ALEXNET-NO_AUG",
    "ALEXNET-AUG",
    "RESNET18-NO_AUG",
    "RESNET18-AUG",
]

# s1 - len 4; c1 - len 3; s2 - len 3; c2 - len 1?; s2b - len 3; c2b_seq - single tensor; c2b_score - single tensor; s3 - len 1; global_pool - single tensor

layers_to_eval = {
    'CHRESMAX_V3_2': ['model_backbone.s1', 'model_backbone.c1', 'model_backbone.s2', 'model_backbone.c2', 'model_backbone.s2b', 'model_backbone.c2b_seq', 'model_backbone.c2b_score', 'model_backbone.s3', 'model_backbone.global_pool'],
    'HMAX_V3_ADJ': ['model_backbone.s1', 'model_backbone.c1', 'model_backbone.s2', 'model_backbone.c2', 'model_backbone.s2b', 'model_backbone.c2b_seq', 'model_backbone.c2b_score', 'model_backbone.s3', 'model_backbone.global_pool'],
    'RESNET50': ['layer1', 'layer2', 'layer3', 'layer4'],
    'RESNET18-AUG': ['layer1', 'layer2', 'layer3', 'layer4'],
}

# Pasupathy data directory
PASUPATHY_DATA_DIR = "/oscar/data/tserre/xyu110/subplots"
OUTPUT_DIR = "./pasupathy_results_1111"

def evaluate_model_on_pasupathy(model_name, layer_name=None, use_neuron_analysis=True):
    """Evaluate a single model on Pasupathy experiment using enhanced analysis"""

    model = get_model(model_name).to(device).eval()
    
    imgsize = 322
    if model_name in baselines:
        imgsize = 227
    

    layer_names = get_all_layer_names(model)
        
    rfanalyzer = RFAnalyzer(enable_upper_bound=True)
    rf_analysis_results = rfanalyzer.analyze_model(model, input_size=imgsize)
    rf_dict = {result['name']: result['rf'][0] for result in rf_analysis_results}

    # get intersection of layer_names and rf_layer_names
    rf_layer_names = [x['name'] for x in rf_analysis_results]
    layer_names = [ln for ln in layer_names if ln in rf_layer_names]
    
    print(layer_names)
    
    if model_name in layers_to_eval:
        layer_names = layers_to_eval[model_name]
        
    if layer_name is None:
        print(f"Will evaluate {len(layer_names)} layers for {model_name} using enhanced analysis")
    else:
        print(f"Evaluating only specified layer: {layer_name}")
        layer_names = [layer_name]

    results = []
    
    # Set up results file paths
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    results_file = os.path.join(OUTPUT_DIR, "all_pasupathy_scores.csv")
    
    # Create subdirectories for detailed results
    neuron_analysis_dir = os.path.join(OUTPUT_DIR, f"neuron_analysis_{model_name}")
    os.makedirs(neuron_analysis_dir, exist_ok=True)
    
    for layer in layer_names:
        try:
            print(f"Evaluating {model_name} on layer: {layer}")
            rf_size = rf_dict[layer]

            # Set up Pasupathy experiment
            pasupathy_exp = Pasupathy(
                model=model, 
                outdir=neuron_analysis_dir, 
                device=device, 
                data_dir=PASUPATHY_DATA_DIR, 
                rf_size=rf_size,
                img_size=imgsize, 
                layer=layer,
                curv_sets=[1, 2],  # Default curvature sets
                rotations=list(range(1, 8)),  # Rotations 1-7
                scales=[0.4, 0.6, 0.8, 1.0]  # Default scales
            )
            
            if use_neuron_analysis:
                print(f"Running comprehensive neuron analysis for {layer}...")
                comprehensive_results = pasupathy_exp.run_neuron_analysis()
                
                pasupathy_exp.save_comprehensive_results(comprehensive_results)
                
                # Extract key metrics
                summary_stats = comprehensive_results['summary_statistics']
                pasupathy_score = summary_stats['slope_mean']  # Main Pasupathy score
                total_neurons = summary_stats['total_neurons']
                r_squared_mean = summary_stats['r_squared_mean']
                slope_std = summary_stats['slope_std']
                
                print(f"Pasupathy Score (mean slope) for {model_name}, {layer}: {pasupathy_score:.4f}")
                print(f"Total neurons analyzed: {total_neurons}")
                print(f"Mean R-squared: {r_squared_mean:.4f}")
                print(f"Score standard deviation: {slope_std:.4f}")
                
                # Store detailed results
                results.append((model_name, layer, pasupathy_score, total_neurons, r_squared_mean, slope_std))
                
                if not os.path.exists(results_file):
                    with open(results_file, "w") as f:
                        f.write("model,layer,pasupathy_score,total_neurons,r_squared_mean,slope_std\n")

                # Save result to file with additional metrics
                with open(results_file, "a") as f:
                    f.write(f"{model_name},{layer},{pasupathy_score:.4f},{total_neurons},{r_squared_mean:.4f},{slope_std:.4f}\n")
                
                # Create a summary file for this specific model-layer combination
                clean_layer_name = layer.replace('.', '_').replace('/', '_')
                # create folder for summary
                neuron_analysis_dir_summary = os.path.join(neuron_analysis_dir, "summaries")
                os.makedirs(neuron_analysis_dir_summary, exist_ok=True)
                summary_file = os.path.join(neuron_analysis_dir_summary, f"{clean_layer_name}_summary.txt")
                
                with open(summary_file, "w") as f:
                    f.write(f"Pasupathy Analysis Summary\n")
                    f.write(f"=" * 50 + "\n")
                    f.write(f"Model: {model_name}\n")
                    f.write(f"Layer: {layer}\n")
                    f.write(f"Image size: {imgsize}\n")
                    f.write(f"\nKey Results:\n")
                    f.write(f"  Pasupathy Score (mean slope): {pasupathy_score:.4f} ± {slope_std:.4f}\n")
                    f.write(f"  Total neurons analyzed: {total_neurons}\n")
                    f.write(f"  Mean R-squared (fit quality): {r_squared_mean:.4f} ± {summary_stats['r_squared_std']:.4f}\n")
                    f.write(f"  Median scale invariance score: {summary_stats['slope_median']:.4f}\n")
                    f.write(f"  Mean maximum activity: {summary_stats['max_activity_mean']:.4f} ± {summary_stats['max_activity_std']:.4f}\n")
                    f.write(f"\nPreferred Rotation Distribution:\n")
                    for rotation, count in summary_stats['preferred_rotations'].items():
                        percentage = (count / total_neurons) * 100
                        f.write(f"  Rotation {rotation}: {count} neurons ({percentage:.1f}%)\n")
                    f.write(f"\nFiles generated:\n")
                    f.write(f"  - Histogram plots: {model_name}_{clean_layer_name}_histograms.png\n")
                    f.write(f"  - Complete results: {model_name}_{clean_layer_name}_results.json/pkl\n")
                    f.write(f"  - Summary statistics: {model_name}_{clean_layer_name}_stats.csv\n")
                
                print(f"Analysis summary saved to: {summary_file}")
                
            # Clean up GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            # Re-raise BdbQuit to allow proper debugger exit
            import bdb
            if isinstance(e, bdb.BdbQuit):
                raise e
            
            print(f"Error evaluating {model_name} on layer {layer}: {e}")
            
            import traceback
            print("Full traceback:")
            traceback.print_exc()  # This will show the exact line

            results.append((model_name, layer, "FAILED", "N/A", "N/A", "N/A"))
            if not os.path.exists(results_file):
                with open(results_file, "w") as f:
                    f.write("model,layer,pasupathy_score,total_neurons,r_squared_mean,slope_std\n")
            with open(results_file, "a") as f:
                f.write(f"{model_name},{layer},FAILED,N/A,N/A,N/A\n")
            # Clean up GPU memory even on error
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            continue

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate HMAX models on Pasupathy experiment")
    parser.add_argument('--model', type=str, help="Specific model to evaluate (optional)")
    parser.add_argument('--layer', type=str, default=None, help="Specific layer to evaluate (optional)")

    args = parser.parse_args()
    
    # python evaluate_hmax_pasupathy.py --model HMAX_V3_ADJ --layer model_backbone.c1
    
    if args.model:
        # Evaluate specific model
        if args.model not in hmax_models:
            print(f"Error: {args.model} not in available models: {hmax_models}")
            exit(1)
        
        print(f"Evaluating specific model: {args.model}")
        print(f"Using enhanced Pasupathy analysis with neuron analysis")
        results = evaluate_model_on_pasupathy(args.model, args.layer, use_neuron_analysis=True)
