"""
runs an evaluation based on the Hangul characters dataset
"""
import numpy as np
import torch.nn as nn
import torch as torch
from torchvision import transforms,datasets

torch.manual_seed(1)
np.random.seed(1)

import os
import csv
import random
import argparse
from tqdm import tqdm
 

from timm.models import create_model

class FeatureExtractor(nn.Module):
    def __init__(self, model, layers):
        super().__init__()
        self.model = model
        self.layers = layers
        self._features = {layer: torch.empty(0) for layer in layers}

        for layer_id in layers:
            # print(dict([*self.model.named_modules()]).keys())
            layer = dict([*self.model.named_modules()])[layer_id]
            layer.register_forward_hook(self.save_outputs_hook(layer_id))

    def save_outputs_hook(self, layer_id):
        def fn(_, __, output):
            self._features[layer_id] = output
        return fn

    def forward(self, x):
        _ = self.model(x)
        return self._features
    

class Invert:
    def __call__(self, sample):
        inverted_image = (-1 * sample) + 1
        return inverted_image

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
                # print(f"max test accuracy : {max(collect)}")
                # maxes[(target_size, test_size)] = max(collect)
                # print(f"std test accuracy : {statistics.pstdev(collect)}")
                # errs[(target_size, test_size)] = statistics.pstdev(collect)
                
        return means
    
    def get_accuracy_arjun(self, filepaths):
        means = {}
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
                    

                normalized = correlations

                correct = []
                distractor = []
                for i in range(0, 53, 2):
                    correct.append(normalized[i][i])
                    distractor.append(normalized[i][i + 1])

                \
                threshold = np.min(normalized + 0.00001)

                above_threshold = normalized > threshold

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

                above_threshold = normalized > best_threshold

                correctly_above_threshold = sum(i > best_threshold for i in correct)
                incorrectly_above_threshold = sum(i > best_threshold for i in distractor)
                correctly_below_threshold = (27) - incorrectly_above_threshold

                print(f'accuracy: {(correctly_above_threshold + correctly_below_threshold)/(54)}')

                means[(target_size, test_size)] = (correctly_above_threshold + correctly_below_threshold)/(54)
            
        return means

    
    def set_layer(self, layer_name):
        self.layer = layer_name

    def run(self):
        pairs = [(13, 13), (13, 52), (13, 130)]
        filepaths = [os.path.join(self.outdir, f"{size_1}-{size_2}.csv") for (size_1, size_2) in pairs]
        self.create_correlation_matrices_batched(pairs, self.layer)
        accs = self.get_accuracy(filepaths)
        print(accs)
        return accs


def load_chmax(layername=None):
    kwargs = {
        'ip_scale_bands': 18,
        'classifier_input_size': 4096,
        'bypass': True,
        'c_debug': False,
    }
    # "/oscar/data/tserre/xyu110/pytorch-output/train/0/mnist/ip_18_hmax_old_gpu_1_cl_0.5_ip_3_224_224_0000_c1[_6,3,1_]_bypass_1/model_best.pth.tar",
    # /oscar/home/npant1/data/npant1/HMAX-epoch=59-val_acc1=99.36899038461539-val_loss=0.029037245774629693.ckpt
    model = create_model(
        'hmax_old',
        pretrained="/oscar/data/tserre/xyu110/pytorch-output/train/0/mnist/ip_18_hmax_old_gpu_1_cl_0.5_ip_3_224_224_0000_c1[_6,3,1_]_bypass_1/model_best.pth.tar",
        num_classes=10,
        in_chans=3,
        global_pool=None,
        scriptable=False,
        **kwargs
    )
    
    # Set the critical attributes that your friend identified
    model.model_pre.base_scale = 224
    model.model_pre.ip_scales = 18
    
    layers = dict([*model.named_modules()]).keys()
    # filter layers
    layers = [layer for layer in layers]
    print(layers)
    return model, "hmax_old", layername, layers


def load_chresmax_v3_bypass_only(layername=None):
    kwargs = {
        'ip_scale_bands': 11,
        'classifier_input_size': 4096,
        'bypass': True,
        'c_debug': False,
    }
    model = create_model(
        'chresmax_v3_bypass_only',
        pretrained='/oscar/data/tserre/xyu110/pytorch-output/train/0/mnist_new/ip_11_chresmax_v3_bypass_only_gpu_8_cl_0.5_ip_3_224_224_4096_c1[_6,3,1_]_bypass/model_best.pth.tar',
        num_classes=10,
        in_chans=3,
        global_pool=None,
        scriptable=False,
        **kwargs
    )
    model.stream_1_bool = True
    model.load_state_dict(torch.load('/oscar/data/tserre/xyu110/pytorch-output/train/0/mnist_new/ip_11_chresmax_v3_bypass_only_gpu_8_cl_0.5_ip_3_224_224_4096_c1[_6,3,1_]_bypass/model_best.pth.tar', map_location='cpu')['state_dict'], strict=True)
    layers = dict([*model.named_modules()]).keys()
    # filter layers
    layers = [layer for layer in layers]
    print(layers)
    return model, 'chresmax_v3_bypass_only', layername, layers



def load_models(modelname, layername=None):
    if modelname == 'hmax_old':
        return load_chmax(layername)
    elif modelname == 'chresmax_v3_bypass_only':
        # python korean.py --model_name chresmax_v3_bypass_only --layer_name model_backbone.s2b
        return load_chresmax_v3_bypass_only(layername)
    else:
        raise ValueError(f"Unknown model name: {modelname}")

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Hangul character evaluation for a single model layer.")
    parser.add_argument('--model_name', type=str, default="hmax_old_original", choices=['hmax_old_arjun', 'hmax_old_original', 'hmax_old', 'chresmax_v3_bypass_only', 'chresmax_v3_bypass_only_c2b', 'chresmax_abs_bypass_only', 'hmax_new_tricks'], help='The name of the model to load.')
    parser.add_argument('--layer_name', type=str, default="model_pre.c2b", help='The specific layer to evaluate.')
    parser.add_argument('--run_s2b_all_layers', action='store_true', help='Run the evaluation for all layers in the S2B model.')

    args = parser.parse_args()
    
    
    layer_to_process = args.layer_name
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    
    model, modelname, _, _ = load_models(args.model_name)
    model = model.to(device)
    # print(vars(model))
    print(f"Loaded model: {modelname} with layer: {layer_to_process}")
    
    korean = Korean(model,
                os.path.join('/oscar/data/tserre/xyu110/pytorch-output/korean', modelname),
                device,
                '/gpfs/data/tserre/npant1/hangul_data',
                224,
                layer_to_process)
    
    
    if args.run_s2b_all_layers:
        
        all_results = []
        
        for index in range(11):
            korean.feature_index = index
            accs = korean.run()
            
            all_results.append([layer_to_process, index, accs])
        
        # Write all results to CSV
        with open("./test.csv", 'a') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['layer', 'index', 'accs'])
            writer.writerows(all_results)
            
        from collections import defaultdict
        values_by_key = defaultdict(list)

        for data_dict in [result[2] for result in all_results]:
            for key, value in data_dict.items():
                values_by_key[key].append(value)
        
        accs = {key: np.max(value_list) for key, value_list in values_by_key.items()}
        
    else:
        
        try:
            accs = korean.run()
            
            
        except Exception as e:
            print(f"Error running Korean experiment for layer {layer_to_process}: {e}")
            # write error to txt file
            with open(os.path.join(korean.outdir, 'error_log.txt'), 'a') as f:
                f.write(f"Error running Korean experiment for layer {layer_to_process}: {e}\n")
    
    # write results
    with open(os.path.join('/oscar/data/tserre/xyu110/pytorch-output/korean', f"results.csv"), 'a') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([modelname, layer_to_process, accs])

