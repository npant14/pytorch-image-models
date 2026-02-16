"""
runs an evaluation based on the Hangul characters dataset
"""
import numpy as np
import torch.nn as nn
import torch as torch
from torchvision import transforms,datasets

torch.manual_seed(1)
np.random.seed(1)

import sys
import os

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

import csv
import random
import argparse
from tqdm import tqdm

from timm.models import create_model
from timm.models.RESMAX import chresmax_v3_2_abs, chresmax_v3_2, hmax_v3_adj
from timm.models.alexnet import alexnet
from timm.models.resnet import resnet18
from utils_hmax import (FeatureExtractor, Invert, load_vit_base, load_hmax_v3_adj, 
                        load_resnet_with_aug, load_alexnet_with_aug, 
                        load_chresmax_v3_2, load_chresmax_v3_2_abs, load_chmax)
    

class korean_dataloader():
    def __init__(self, image_size, datadir):
        super().__init__()

        # Directory to load Data
        self.datadir = datadir
        self.image_size = image_size
        self.data = datasets.ImageFolder(root=
            self.datadir,
            transform=
            transforms.Compose([
                transforms.ToTensor(),
                Invert(),
                transforms.ToPILImage(),
                transforms.Resize((40,40)),
                transforms.Pad(((self.image_size - 40)//2, (self.image_size - 40)//2)),
                transforms.ToTensor(),
            ]))

    def __getitem__(self, idx):
        img = self.data[idx]
        return img

    def __len__(self):
        return len(self.data)
    


class Korean():
    def __init__(self, model, outdir, device, data_dir, img_size=322, layer = "s3.layer.3.conv3"):
        self.model = model
        self.outdir = os.path.join(outdir, layer)
        self.device = device
        self.data_dir = data_dir
        self.img_size = img_size
        self.layer = layer
        self.data = korean_dataloader(img_size, data_dir)
        # self.data = get_korean_dataloader_arjun(img_size, 1, 1)
        
        self.feature_index = 0
        
        os.makedirs(self.outdir, exist_ok=True)
        print("setup Korean experiment -- ready to run")

    '''
    This function is expecting a square image that has a character 
    that is 40x40 pixels in the center of the image

    1 < size < image.shape[1]
    size is the new size of the character
    '''
    def resize_image(self, image, size):
        image_size = image.shape[1]
        pil_image = transforms.ToPILImage()(image)
        pad_before = (image_size - size)//2
        pad_after = (image_size - size)//2
        if pad_after + pad_before + size != image_size:
            pad_after += (image_size - (pad_after + pad_before + size))
        resize = transforms.Compose([
                            transforms.CenterCrop(40),
                            transforms.Resize((size,size)),
                            transforms.Pad((pad_before, pad_before, pad_after, pad_after)),
                            transforms.ToTensor()
                        ])
        
        new_img = resize(pil_image)

        return new_img

    def compute_dprime(self, correct_correlations, distractor_correlations, threshold):
        """
        d' = (proportion correct on target/distractor pairs) - (proportion incorrect on target/target pairs)
        """
        # Create binary arrays based on threshold
        target_distractor_pairs = [1 if i > threshold else 0 for i in correct_correlations]
        target_target_pairs = [1 if i <= threshold else 0 for i in distractor_correlations]
        
        target_distractor_correct = np.mean(target_distractor_pairs)
        target_target_incorrect = 1.0 - np.mean(target_target_pairs)
        
        d_prime = target_distractor_correct - target_target_incorrect
        
        return d_prime
    
    def get_pearson_correlation(self, tensor_1, tensor_2):
        '''
        returns the pearson correlation for a pair of tensors
        '''
        tensor_features = torch.stack((tensor_1, tensor_2))
        return torch.corrcoef(tensor_features)[0][1].item()

    def create_correlation_matrices(self, pairs, layer_name):
        '''
        parameters:
        pairs -- the pairs of sizes to calculate correlations for 
                expects list of tuples (ex [(20,20), (20,80), (20,200)])
        '''
        activations_dict = {}
        correlation_matix = np.zeros((len(self.data), len(self.data)))

        for size_1, size_2 in pairs:
            print(f"running {size_1}, {size_2}")
            for i, im1 in enumerate(tqdm(self.data, desc=f"Row Images ({size_1})")):
                for j, im2 in enumerate(tqdm(self.data, desc=f"Col Images ({size_2})", leave=False)):
                    
                    layer_features = FeatureExtractor(self.model, [layer_name])
                    resized_img = self.resize_image(im1[0], size_1)
                    features = layer_features(torch.unsqueeze(resized_img, 0).to(self.device))
                    tensor_feature = features[layer_name]
                    # old hmax go deeper
                    if type(tensor_feature) is tuple or type(tensor_feature) is list:
                        tensor_feature = features[layer_name][0][0]
                        
                    # import pdb; pdb.set_trace()
                    rowfeat = torch.squeeze(torch.flatten(tensor_feature))


                    layer_features = FeatureExtractor(self.model, [layer_name])
                    resized_img = self.resize_image(im2[0], size_2)
                    features = layer_features(torch.unsqueeze(resized_img, 0).to(self.device))
                    tensor_feature = features[layer_name]
                    # old hmax go deeper
                    if type(tensor_feature) is tuple or type(tensor_feature) is list:
                        tensor_feature = features[layer_name][0][0]
                    colfeat = torch.squeeze(torch.flatten(tensor_feature))

                    ij_corr = self.get_pearson_correlation(rowfeat, colfeat)
                    correlation_matix[i][j] = ij_corr
                    del layer_features
                    del features
                    del rowfeat
                    del colfeat


            with open(os.path.join(self.outdir, f"{size_1}-{size_2}.csv"), 'w') as f:
                print(f"writing out {size_1} {size_2} to csv")
                writer = csv.writer(f)
                for row in correlation_matix:
                    writer.writerow(row)
                    
    def create_correlation_matrices_batched(self, pairs, layer_name, batch_size=4):
        '''
        Batched version of create_correlation_matrices for better GPU utilization
        parameters:
        pairs -- the pairs of sizes to calculate correlations for 
                expects list of tuples (ex [(20,20), (20,80), (20,200)])
        layer_name -- the layer to extract features from
        batch_size -- number of images to process in each batch
        '''
        
        def get_features_batch(images, size):
            """Process a batch of images and return their features"""
            batch_imgs = []
            for img in images:
                resized_img = self.resize_image(img, size)
                batch_imgs.append(resized_img)
            
            # Stack images into a batch tensor
            batch_tensor = torch.stack(batch_imgs).to(self.device)
            
            # Extract features for the batch
            layer_features = FeatureExtractor(self.model, [layer_name])
            with torch.no_grad():  # Save memory by disabling gradients
                features = layer_features(batch_tensor)
                tensor_feature = features[layer_name]
                
                # Handle old hmax structure
                if type(tensor_feature) is tuple or type(tensor_feature) is list:
                    tensor_feature = tensor_feature[self.feature_index][0]
                
                # Flatten each feature vector and move to CPU
                flattened = torch.flatten(tensor_feature, start_dim=1)
                batch_features = [feat.cpu().clone() for feat in flattened]
            
            del layer_features, features, tensor_feature, batch_tensor
            torch.cuda.empty_cache()
            
            return batch_features
        
        # Pre-compute features for all sizes
        features_cache = {}
        unique_sizes = set()
        for size_1, size_2 in pairs:
            unique_sizes.add(size_1)
            unique_sizes.add(size_2)
        
        print("Pre-computing features...")
        for size in unique_sizes:
            print(f"Computing features for size {size}")
            size_features = []
            
            # Extract all images from the dataset
            all_images = [self.data[i][0] for i in range(len(self.data))]
            
            # Process in batches
            for i in range(0, len(all_images), batch_size):
                end_idx = min(i + batch_size, len(all_images))
                batch_imgs = all_images[i:end_idx]
                
                try:
                    batch_features = get_features_batch(batch_imgs, size)
                    size_features.extend(batch_features)
                    print(f"Processed batch {i//batch_size + 1}/{(len(all_images) + batch_size - 1)//batch_size}")
                    
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print(f"OOM in batch, processing individually...")
                        torch.cuda.empty_cache()
                        # Process one by one
                        for single_img in batch_imgs:
                            single_features = get_features_batch([single_img], size)
                            size_features.extend(single_features)
                    else:
                        raise e
            
            features_cache[size] = size_features
            print(f"Cached {len(size_features)} features for size {size}")
        
        # Now compute correlations using cached features
        for size_1, size_2 in pairs:
            print(f"Computing correlations for {size_1}, {size_2}")
            correlation_matrix = np.zeros((len(self.data), len(self.data)))
            
            features_1 = features_cache[size_1]
            features_2 = features_cache[size_2]
            
            for i in tqdm(range(len(self.data)), desc=f"Row Images ({size_1})"):
                for j in range(len(self.data)):
                    # Move features to GPU only when computing correlation
                    rowfeat = features_1[i].to(self.device)
                    colfeat = features_2[j].to(self.device)
                    
                    ij_corr = self.get_pearson_correlation(rowfeat, colfeat)
                    correlation_matrix[i][j] = ij_corr
                    
                    # Clean up
                    del rowfeat, colfeat

            # Save results
            with open(os.path.join(self.outdir, f"{size_1}-{size_2}.csv"), 'w') as f:
                print(f"writing out {size_1} {size_2} to csv")
                writer = csv.writer(f)
                for row in correlation_matrix:
                    writer.writerow(row)
        
        print("All correlations computed!")


    def get_accuracy(self, filepaths):
        means = {}
        d_primes = {}
        errs = {}
        maxes = {}
        # iterate through all the saved csvs
        for path in tqdm(filepaths, desc="Evaluating CSV Accuracy"):
            if path.split("/")[-1].startswith('2scale-'):
                target_size, test_size = path.replace('2scale-', '').split("/")[-1].split(".")[0].split("-")
            else:
                target_size, test_size = path.split("/")[-1].split(".")[0].split("-")
            # want to check both directions
            for transpose in [False, True]:
                CSVData = open(path)
                correlations = np.loadtxt(CSVData, delimiter=",")

                if transpose:
                    correlations = correlations.transpose()
                    target_size, test_size = test_size, target_size

                # normalization was wrong, did a double normalization
                normalized = correlations
                # normalized = (correlations - np.min(correlations, axis=0)) / (np.max(correlations) - np.min(correlations))

                all_correct = []
                all_distractor = []
                for i in range(0, 53, 2):
                    all_correct.append(normalized[i][i])
                    all_distractor.append(normalized[i][i + 1])

                collect = []

                for _ in range(1000):
                    randidxs = random.sample(range(54), k=41)
                    randidxs.sort()

                    thresh_correct = []
                    test_correct = []
                    thresh_distractor = []
                    test_distractor = []
                    idx = 0

                    for idx, val in enumerate(all_correct):
                        if idx in randidxs:
                            thresh_correct.append(val)
                        else:
                            test_correct.append(val)

                    for idx, val in enumerate(all_distractor):
                        if idx+27 in randidxs:
                            thresh_distractor.append(val)
                        else:
                            test_distractor.append(val)

                    correct = thresh_correct
                    distractor = thresh_distractor


                    best_threshold = 0
                    best_accuracy = 0

                    for thresh in correct + distractor:
                        correctly_above_threshold = sum(i > thresh for i in correct)
                        incorrectly_above_threshold = sum(i > thresh for i in distractor)
                        correctly_below_threshold = (27) - incorrectly_above_threshold
                        acc = (correctly_above_threshold + correctly_below_threshold)/(54)
                        if acc >= best_accuracy:
                            best_accuracy = acc
                            best_threshold = thresh
                    
                    # print(f"best threshold : {best_threshold}")
                    # print(f"best accuracy : {best_accuracy}")

                    test_correctly_above_threshold = sum(i > best_threshold for i in correct + test_correct)
                    test_incorrectly_above_threshold = sum(i > best_threshold for i in distractor + test_distractor)
                    test_correctly_below_threshold = (27) - test_incorrectly_above_threshold
                    test_acc = (test_correctly_above_threshold + test_correctly_below_threshold)/(54)

                    collect.append(test_acc)

                print(f"average test accuracy : {sum(collect)/len(collect)}")
                means[(target_size, test_size)] = sum(collect)/len(collect)
                
                # Calculate d-prime
                d_prime = self.compute_dprime(correct, distractor, best_threshold)
                d_primes[(target_size, test_size)] = d_prime
                print(f'd-prime: {d_prime}')
                
        return means, d_primes
    
    def set_layer(self, layer_name):
        self.layer = layer_name

    def run(self):
        pairs = [(13, 13), (13, 52), (13, 130)]
        filepaths = [os.path.join(self.outdir, f"{size_1}-{size_2}.csv") for (size_1, size_2) in pairs]
        self.create_correlation_matrices_batched(pairs, self.layer)
        accs, d_primes = self.get_accuracy(filepaths)
        print("Accuracies:", accs)
        print("D-primes:", d_primes)
        return accs, d_primes


def load_models(modelname, layername=None):
    if modelname == 'chresmax_v3_2':
        model = load_chresmax_v3_2(device=device)
        img_size = 322  # chresmax_v3_2 uses 322x322
    elif modelname == 'chresmax_v3_abs':
        model = load_chresmax_v3_2_abs(device=device)
        img_size = 322  # chresmax_v3_2_abs uses 322x322
    elif modelname == 'alexnet':
        model = load_alexnet_with_aug(device=device)
        img_size = 227  # AlexNet uses 227x227
    elif modelname == 'resnet18':
        model = load_resnet_with_aug(device=device)
        img_size = 227  # ResNet uses 227x227
    elif modelname == 'hmax_v3_adj':
        model = load_hmax_v3_adj(device=device)
        img_size = 322  # HMAX uses 322x322
    elif modelname == 'vit_base':
        model = load_vit_base(device=device)
        img_size = 224  # ViT uses 224x224
    else:
        raise ValueError(f"Unknown model name: {modelname}")

    layers = dict([*model.named_modules()]).keys()
    print(layers)
    
    return model, modelname, layername, layers, img_size

    
def run_all_layers_experiment(model, modelname, all_layer_names, device, img_size, output_dir):
    """Run Korean experiment for all layers in the model."""
    print(f"Running experiment for all {len(all_layer_names)} layers with image size {img_size}")
    
    experiment_results = []
    
    for current_layer in all_layer_names:
        try:
            korean_experiment = Korean(
                model,
                os.path.join(output_dir, modelname),
                device,
                '/gpfs/data/tserre/npant1/hangul_data',
                img_size,
                current_layer
            )
            
            layer_accuracies, layer_d_primes = korean_experiment.run()
            experiment_results.append([current_layer, layer_accuracies, layer_d_primes])
            print(f"Completed layer {current_layer}: accuracies={layer_accuracies}, d_primes={layer_d_primes}")
            
        except Exception as e:
            print(f"Error running Korean experiment for layer {current_layer}: {e}")
            # Store default error result to maintain data structure
            default_error_result = {
                ('13', '13'): 0.0, ('13', '52'): 0.0, ('13', '130'): 0.0,
                ('52', '13'): 0.0, ('130', '13'): 0.0,
            }
            experiment_results.append([current_layer, default_error_result, default_error_result])
    
    # Save all results to CSV in the output directory
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{modelname}_all_layers.csv")
    with open(output_file, 'a') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['layer', 'accuracies', 'd_primes'])
        writer.writerows(experiment_results)
    
    print(f"All layer results saved to {output_file}")
    return experiment_results


def run_single_layer_experiment(model, modelname, target_layer, device, img_size, output_dir):
    """Run Korean experiment for a single specified layer."""
    print(f"Running single layer experiment for: {target_layer} with image size {img_size}")
    
    try:
        korean_experiment = Korean(
            model,
            os.path.join(output_dir, modelname),
            device,
            '/gpfs/data/tserre/npant1/hangul_data',
            img_size,
            target_layer
        )
        
        accuracies, d_primes = korean_experiment.run()
        print(f"Single layer results - accuracies: {accuracies}")
        print(f"Single layer results - d_primes: {d_primes}")
        return {'accuracies': accuracies, 'd_primes': d_primes}
        
    except Exception as e:
        print(f"Error running Korean experiment for layer {target_layer}: {e}")
        
        # Log error to file if possible
        try:
            error_log_path = os.path.join(korean_experiment.outdir, 'error_log.txt')
            with open(error_log_path, 'a') as f:
                f.write(f"Error running Korean experiment for layer {target_layer}: {e}\n")
        except:
            print("Could not write error log file")
        
        return None


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run Hangul character evaluation for model layers.")
    parser.add_argument('--model_name', type=str, default="hmax_v3_adj",
                       help='The name of the model to load.')
    parser.add_argument('--layer_name', type=str, default="model_pre.c2b",
                       help='The specific layer to evaluate (for single layer mode).')
    parser.add_argument('--run_all_layers', action='store_true',
                       help='Run evaluation for all layers in the model.')
    parser.add_argument('--output_dir', type=str, default="results/korean_results_dprime",
                       help='Directory to save results (default: results/korean_results_dprime)')

    args = parser.parse_args()
    
    # Setup device and load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, model_name, _, available_layers, img_size = load_models(args.model_name)
    model = model.to(device)
    
    print(f"Loaded model: {model_name}")
    print(f"Output directory: {args.output_dir}")
    print(f"Available layers: {len(available_layers)}")
    print(f"Image size: {img_size}")
    
    # Run the appropriate experiment based on arguments
    final_results = None
    
    if args.run_all_layers:
        final_results = run_all_layers_experiment(model, model_name, available_layers, device, img_size, args.output_dir)
        
    else:
        final_results = run_single_layer_experiment(model, model_name, args.layer_name, device, img_size, args.output_dir)
    
    # Write final results to master results file (only if we have results)
    if final_results is not None:
        os.makedirs(args.output_dir, exist_ok=True)
        master_results_path = os.path.join(args.output_dir, "results.csv")
        with open(master_results_path, 'a') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([model_name, args.layer_name, final_results])
        print(f"Final results written to {master_results_path}")
    else:
        print("No results to write - experiment failed")




