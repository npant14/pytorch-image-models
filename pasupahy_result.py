

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import torch

torch.manual_seed(1)
np.random.seed(1)

from timm.models.RESMAX import hmax_v3_adj
from timm.models.alexnet import alexnet
from timm.models.resnet import resnet18

plt.rcParams['figure.figsize'] = (14, 10)

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_alexnet(layername=None):
    # checkpoint_path = "/oscar/data/tserre/xyu110/pytorch-output/train/0/baseline_w_aug/ip_0_alexnet_gpu_2_cl_0_ip_3_227_227_0_c1[_6,3,1_]_scale_0.08/model_best.pth.tar"
    checkpoint_path = "/oscar/data/tserre/xyu110/pytorch-output/train/sep/alexnet_fair_comparasion/model_best.pth.tar"
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = alexnet(channel_size=227).to(device).eval()
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    layers = dict([*model.named_modules()]).keys()
    print(layers)
    return model, 'alexnet', layername, layers


def load_resnet18(layername=None):
    # checkpoint_path = "/oscar/data/tserre/xyu110/pytorch-output/train/0/baseline_w_aug/ip_0_resnet18_gpu_8_cl_0_ip_3_227_227_512_c1[_6,3,1_]_scale_0.08/model_best.pth.tar"
    checkpoint_path = "/oscar/data/tserre/xyu110/pytorch-output/train/sep/resnet_18_fair_comparasion/model_best.pth.tar"
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = resnet18(channel_size=227).to(device).eval()
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    layers = dict([*model.named_modules()]).keys()
    print(layers)
    return model, 'resnet18', layername, layers


def load_hmax_v3_adj(layername=None):
    checkpoint_path = '/oscar/data/tserre/xyu110/pytorch-output/train/0/final_versions/ip_3_hmax_v3_adj_gpu_8_cl_0.1_ip_3_322_322_18432_c1[_6,3,1_]_bypass/model_best.pth.tar'
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = hmax_v3_adj().to(device).eval()
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    layers = dict([*model.named_modules()]).keys()
    print(layers)
    return model, 'hmax_v3_adj', layername, layers

# %%
base = "/users/xyu110/pytorch-image-models/pasupathy_results_1110"
alexnet_data_dir = f"{base}/neuron_analysis_ALEXNET-AUG/neuron_data"
resnet18_data_dir = f"{base}/neuron_analysis_RESNET18-AUG/neuron_data"
hmax_v3_adj_data_dir = f"{base}/neuron_analysis_HMAX_V3_ADJ/neuron_data"

# Function to load layer data from a directory
def load_layer_data(data_dir, layers):
    model_layer_data = {}
    for layer_name in layers:
        safe_layer_name = layer_name.replace('.', '_')
        file_path = Path(data_dir) / f"{safe_layer_name}.csv"
        if file_path.exists():
            df = pd.read_csv(file_path)
            if 'scale_invariance_score' in df.columns:
                scores = df['scale_invariance_score'].dropna()
                if len(scores) > 0:
                    model_layer_data[layer_name] = scores.values
    return dict(model_layer_data.items())

# Load AlexNet and ResNet18 layers
_, _, _, layers_alexnet = load_alexnet()
_, _, _, layers_resnet18 = load_resnet18()
_, _, _, layers_hmax_v3_adj = load_hmax_v3_adj()

alexnet_layer_data = load_layer_data(alexnet_data_dir, layers_alexnet)
resnet18_layer_data = load_layer_data(resnet18_data_dir, layers_resnet18)
hmax_v3_adj_layer_data = load_layer_data(hmax_v3_adj_data_dir, layers_hmax_v3_adj)

print(f"Loaded AlexNet layers: {list(alexnet_layer_data.keys())}")
print(f"Loaded ResNet18 layers: {list(resnet18_layer_data.keys())}")
print(f"Loaded HMAX V3 ADJ layers: {list(hmax_v3_adj_layer_data.keys())}")

# %%
# Calculate median of absolute values for each layer across all models
def calculate_median_abs_scores(layer_data):
    """
    Calculate median of absolute values of scale invariance scores for each layer.
    
    Args:
        layer_data: Dictionary with layer names as keys and arrays of scores as values
    
    Returns:
        dict: Dictionary with layer names as keys and median absolute values as values
    """
    median_abs_dict = {}
    
    for layer_name, scores in layer_data.items():
        abs_scores = np.abs(scores)
        median_abs = np.median(abs_scores)
        median_abs_dict[layer_name] = median_abs
    
    return median_abs_dict

layers_to_eval = {
    'HMAX_V3_ADJ': ['model_backbone.s1', 'model_backbone.c1', 'model_backbone.s2', 'model_backbone.c2', 'model_backbone.s2b', 'model_backbone.c2b_seq', 'model_backbone.c2b_score', 'model_backbone.s3', 'model_backbone.global_pool'],
    'RESNET50': ['layer1', 'layer2', 'layer3', 'layer4'],
    'RESNET18-AUG': ['layer1', 'layer2', 'layer3', 'layer4'],
}
# Calculate median absolute values for all models
alexnet_median_abs = calculate_median_abs_scores(alexnet_layer_data)
resnet18_median_abs = calculate_median_abs_scores(resnet18_layer_data)
hmax_v3_adj_median_abs = calculate_median_abs_scores(hmax_v3_adj_layer_data)

alexnet_median = list(alexnet_median_abs.values())
resnet18_median = list(resnet18_median_abs.values())
hmax_v3_adj_median = list(hmax_v3_adj_median_abs.values())

alexnet_labels = list(alexnet_median_abs.keys())
resnet18_labels = list(layers_to_eval['RESNET18-AUG'])
hmax_v3_adj_labels = list(layers_to_eval['HMAX_V3_ADJ'])

print("Best (lowest) alexnet score: ", f"{min(alexnet_median):.4f}")
print("Best (lowest) resnet18 score: ", f"{min(resnet18_median):.4f}")
print("HMAX V4 scores: S2 - ", f"{hmax_v3_adj_median[2]:.4f}", " C2 - ", f"{hmax_v3_adj_median[3]:.4f}")

# %%
# Combined comparison plot with normalized layer progression
fig, ax = plt.subplots(figsize=(14, 8))

# Find the maximum number of layers to normalize all models to the same width
max_layers = max(len(alexnet_median), len(resnet18_median), len(hmax_v3_adj_median))

# Create normalized x-coordinates for each model (stretched to same width)
alexnet_x = np.linspace(0, max_layers - 1, len(alexnet_median))
resnet18_x = np.linspace(0, max_layers - 1, len(resnet18_median))
hmax_x = np.linspace(0, max_layers - 1, len(hmax_v3_adj_median))

# Plot each model with normalized x-axis (stretched to same width)
ax.plot(alexnet_x, alexnet_median, 
        marker='o', linestyle='-', linewidth=2, markersize=8, 
        label='AlexNet', color='#036AA6')

ax.plot(resnet18_x, resnet18_median, 
        marker='s', linestyle='--', linewidth=2, markersize=8,
        label='ResNet18', color='#ff841f')

ax.plot(hmax_x, hmax_v3_adj_median, 
        marker='d', linestyle='-.', linewidth=2, markersize=8,
        label='HMAX V3 ADJ', color='#279C3A')

# Customize plot
ax.set_ylabel('Median Absolute Scale Invariance Score', fontweight='bold', fontsize=12)
ax.set_title('Comparison of Scale Invariance Across Model Layers', fontweight='bold', fontsize=14)
ax.legend(loc='best', fontsize=11, framealpha=0.9)
ax.grid(True, alpha=0.3, linestyle='--')

# Add x-axis ticks showing only first and last layer
tick_positions = [0, max_layers - 1]  # Only first and last positions
tick_labels = ['First Layer', 'Last Layer']
ax.set_xticks(tick_positions)
ax.set_xticklabels(tick_labels)
ax.set_xlabel('Relative Network Depth', fontweight='bold', fontsize=12)

plt.tight_layout()
plt.savefig('combined_median_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# Print summary statistics
print("\n" + "="*70)
print("SUMMARY: Median Absolute Scale Invariance Scores")
print("="*70)
print(f"AlexNet:      Layers={len(alexnet_median):2d}, Mean={np.mean(alexnet_median):.4f}, Std={np.std(alexnet_median):.4f}")
print(f"ResNet18:     Layers={len(resnet18_median):2d}, Mean={np.mean(resnet18_median):.4f}, Std={np.std(resnet18_median):.4f}")
print(f"HMAX V3 ADJ:  Layers={len(hmax_v3_adj_median):2d}, Mean={np.mean(hmax_v3_adj_median):.4f}, Std={np.std(hmax_v3_adj_median):.4f}")
print("="*70)


