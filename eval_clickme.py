import argparse
import os
import torch
import numpy as np
from torchvision import transforms
from timm.data import resolve_data_config, create_transform

from harmonization.common import load_clickme_val
from harmonization.evaluation import evaluate_clickme

from models.RESMAX import resmax_bypass
from models.ALEXMAX3 import chalexmax_v3_3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -- MODEL LOADERS --
def load_resmax_bypass():
    checkpoint_path = "/cifs/data/tserre_lrs/projects/prj_hmax/models/debug_resmax_bypass_concat_gpu_8_cl_0_ip_3_322_322_18432_c1[_6,3,1_]/model_best.pth.tar"
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model = resmax_bypass(num_classes=1000, big_size=322, small_size=227, classifier_input_size=18432)
    model.load_state_dict(checkpoint['state_dict'])
    return model

def load_chalexmax_v3_3():
    checkpoint_path = "/cifs/data/tserre_lrs/projects/prj_hmax/models/debug4_resize2_chalexmax_v3_3_cl_1_ip_7_322_9216/model_best.pth.tar"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = chalexmax_v3_3(
        num_classes=1000, big_size=322, small_size=227,
        classifier_input_size=9216, ip_scale_bands=7, contrastive_loss=True
    ).to(device)
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    return model

def get_model(model_name):
    if model_name == "CHALEXMAX":
        return load_chalexmax_v3_3()
    elif model_name == "RESMAX":
        return load_resmax_bypass()
    elif model_name == "RESNET50":
        return torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True).to(device)
    #elif model_name=="RESNET18":
        #return model = timm.create_model('resnet18.a1_in1k', pretrained=True) # TODO: Input_size_check 
    elif model_name == "ALEXNET":
        return torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', pretrained=True).to(device)
    else:
        raise ValueError(f"Unsupported model: {model_name}")


def torch_explainer(xbatch, ybatch):
    xbatch = torch.stack([model_preprocess(x) for x in xbatch.numpy().astype(np.uint8)])
    ybatch = torch.tensor(ybatch.numpy(), dtype=torch.long)

    xbatch = xbatch.to(device).requires_grad_()
    ybatch = ybatch.to(device)

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
    parser = argparse.ArgumentParser(description="Evaluate model on ClickMe using torch_explainer.")
    parser.add_argument('--model', type=str, required=True, choices=["CHALEXMAX", "RESMAX", "RESNET50", "ALEXNET"],
                        help="Model to evaluate")
    args = parser.parse_args()

    model = get_model(args.model).to(device).eval()

    # Set up preprocessing
    config = resolve_data_config({'input_size': (3, 322, 322)}, model=model)
    transform = create_transform(**config)
    model_preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((322, 322)),
        transform
    ])

    # Load and resize dataset
    clickme_dataset_resized = prepare_clickme_dataset()

    # Evaluate
    scores = evaluate_clickme(model,
                              explainer=torch_explainer,
                              clickme_val_dataset=clickme_dataset_resized)
    print("Alignment Score:", scores['alignment_score']) 
    # Append result to file
    results_file = "/files22_lrsresearch/CLPS_Serre_Lab/projects/prj_concept_surgery/finetuning_models/model_scores.txt"
    with open(results_file, "a") as f:
        f.write(f"{args.model}: {scores['alignment_score']:.4f}\n") 
