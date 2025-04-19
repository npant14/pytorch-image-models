import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from xplique.attributions import Saliency
from xplique.wrappers import TorchWrapper
import argparse
from clickme_scaling import prepare_clickme_dataset_at_scale
from clickme_scaling import * 
from harmonization.common import load_clickme_val
from timm.data import resolve_data_config, create_transform
from torchvision import transforms
from eval_harm import * 

# === Supported scales ===
SCALES = [160, 192, 227, 270, 322, 382, 454]

# === Directory for visualizations ===
SAVE_DIR = "/media/data_cifs/projects/prj_concept_surgery/finetuning_models/clickme_saliency_overlays"
os.makedirs(SAVE_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import tensorflow as tf
from torchvision import transforms
from xplique.wrappers import TorchWrapper
from xplique.attributions import Saliency
from harmonization.common import load_clickme_val
from clickme_scaling import prepare_clickme_dataset_at_scale
from eval_harm import get_model

SCALES = [160, 192, 227, 270, 322, 382, 454]
NUM_EXAMPLES = 5
SAVE_DIR = "/media/data_cifs/projects/prj_concept_surgery/finetuning_models/clickme_vs_saliency_all_models"
os.makedirs(SAVE_DIR, exist_ok=True)

def overlay_heatmap(img_np, saliency_map, alpha=0.45, cmap='hot'):
    saliency_map = np.squeeze(saliency_map)
    saliency_map -= saliency_map.min()
    if saliency_map.max() > 0:
        saliency_map /= saliency_map.max()
    cmap_img = plt.cm.get_cmap(cmap)(saliency_map)[..., :3]
    if img_np.max() > 1.0:
        img_np = img_np / 255.0
    overlay = img_np * (1 - alpha) + cmap_img * alpha
    return np.clip(overlay, 0, 1)

def visualize_clickme_vs_saliency(clickme_dataset, explainer, model_name, scale):
    for idx, batch in enumerate(clickme_dataset.take(NUM_EXAMPLES)):
        image_tf, heatmap_tf, label_tf = batch[f"images_{scale}"], batch[f"heatmaps_{scale}"], batch['labels']

        image_np = image_tf[0].numpy()
        heatmap_np = heatmap_tf[0].numpy()
        label_np = label_tf[0].numpy()

        image_tensor = torch.tensor(image_np).unsqueeze(0).float()
        #.permute(2, 0, 1)
        label_tensor = torch.tensor([label_np])

        saliency = explainer(image_tensor, label_tensor)[0].numpy()
        saliency_tensor = torch.tensor(saliency).unsqueeze(0).float()
        saliency_tensor.detach().cpu()

        #saliency.permute(1, 2, 0)
        heatmap_np -= heatmap_np.min()
        heatmap_np /= (heatmap_np.max() + 1e-8)
        if image_np.max() > 1.0:
            img_np = image_np / 255.0

        overlay = overlay_heatmap(image_np, saliency)

        fig, axs = plt.subplots(1, 4, figsize=(16, 4))
        axs[0].imshow(img_np)
        axs[0].set_title("Input Image")
        axs[1].imshow(heatmap_np[..., 0], cmap="viridis")
        axs[1].set_title("ClickMe Heatmap")
        axs[2].imshow(saliency.squeeze(), cmap="hot")
        axs[2].set_title("Model Saliency")
        axs[3].imshow(overlay)
        axs[3].set_title("Overlay")
        for ax in axs:
            ax.axis("off")

        plt.tight_layout()
        model_dir = os.path.join(SAVE_DIR, f"{model_name}_scale_{scale}")
        os.makedirs(model_dir, exist_ok=True)
        plt.savefig(os.path.join(model_dir, f"example_{idx}.png"))
        plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs='+', required=True,
                        help='List of model names to evaluate (e.g., CORNET VGGMAX CHALEXMAX)')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for model_name in args.models:
        print(f"\nEvaluating model: {model_name}")
        model = get_model(model_name).to(device).eval()
        wrapped_model = TorchWrapper(model, device=device)
        explainer = Saliency(wrapped_model)

        for scale in SCALES:
            print(f"  Processing scale: {scale}")
            dataset = prepare_clickme_dataset_at_scale(scale)
            visualize_clickme_vs_saliency(dataset, explainer, model_name, scale)

