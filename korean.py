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
import statistics
from tqdm import tqdm
import pytorch_lightning as pl


from timm.models import create_model, load_checkpoint, is_model, list_models

from timm.models.HMAX_old import hmax_old_original
import hmax_fixed_ligtning


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
    
# from torch.utils.data import random_split, DataLoader, Dataset
# class dataa_loader_korean(pl.LightningDataModule):
#     def __init__(self, image_size, traindir, valdir, testdir, batch_size_per_gpu, n_gpus, test_mode = False, \
#                  rdm_corr_mode = False, featur_viz = False, same_scale_viz = False, linderberg_bool = False, \
#                  linderberg_dir = None, linderberg_test = False, orginal_mnist_bool = False):
#         super().__init__()
          
#         # Directory to load Data
#         self.traindir = traindir
#         self.valdir = valdir
#         self.testdir = testdir
#         self.test_mode = test_mode
#         self.rdm_corr_mode = rdm_corr_mode
#         self.featur_viz = featur_viz
#         self.same_scale_viz = same_scale_viz

#         self.image_size = int(image_size)

#     def __getitem__(self, idx):

#         img = self.train_data[idx]

#         return img
 
#     def __len__(self):
#         return len(self.train_data)
    
#     def setup(self, stage=None):

#             self.train_data = datasets.ImageFolder(root=
#                 self.traindir,
#                 transform=
#                 transforms.Compose([
#                     transforms.ToTensor(),
#                     Invert(),
#                     transforms.ToPILImage(),
#                     transforms.Resize((40,40)),
#                     transforms.Pad(((self.image_size - 40)//2, (self.image_size - 40)//2)),
#                     transforms.ToTensor(),
#                 ]))

#             self.val_data = datasets.ImageFolder(root=
#                 self.valdir,
#                 transform=
#                 transforms.Compose([
#                     transforms.Resize((self.image_size, self.image_size)),
#                     transforms.Pad(50),
#                     #transforms.RandomHorizontalFlip(),
#                     transforms.ToTensor(),
#                 ]))

#             if self.test_mode:
#                 self.test_data = datasets.ImageFolder(root=
#                     self.testdir,
#                     transform =
#                     transforms.Compose([
#                     transforms.Resize((self.image_size, self.image_size)),
#                         transforms.ToTensor(),
#                     ]), 
#                     # loader = loader_func 
#                     )


#     def train_dataloader(self):
        
#         # Generating train_dataloader
#         loader = DataLoader(self.train_data, 
#                           batch_size = self.batch_size, drop_last = True, num_workers = 8, pin_memory=False, shuffle = True)
#         return loader
  
#     def val_dataloader(self):
        
#         # Generating val_dataloader
#         return DataLoader(self.val_data,
#                           batch_size = self.batch_size, drop_last = True, num_workers = 8, pin_memory=False, shuffle = True)
  
#     def test_dataloader(self):
        
#         # Generating test_dataloader
#         return DataLoader(self.test_data,
#                           batch_size = self.batch_size, drop_last = True, num_workers = 4, shuffle = False)

# def get_korean_dataloader_arjun(image_size, batch_size_per_gpu, n_gpus):
#     traindir = "/gpfs/data/tserre/npant1/hangul_data"
#     valdir = "/gpfs/data/tserre/npant1/hangul_data"
#     testdir = "/gpfs/data/tserre/npant1/hangul_data"

#     data = dataa_loader_korean(image_size, traindir, valdir, testdir, batch_size_per_gpu, n_gpus)
#     data.setup()

#     return data

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
                    tensor_feature = tensor_feature[0][0]
                
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
                    
                    print(f"best threshold : {best_threshold}")
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


def load_chresmax_v3():
    kwargs = {
        'ip_scale_bands': 11,
        'classifier_input_size': 10496,
        'bypass': True,
        'c_debug': False,
    }
    model = create_model(
        'chresmax_v3',
        pretrained='/oscar/data/tserre/xyu110/pytorch-output/train/mnist/ip_11_chresmax_v3_gpu_2_cl_0.1_ip_3_224_224_10496_c1[_6,3,1_]_bypass_3/checkpoint-10.pth.tar',
        num_classes=10,
        in_chans=3,
        global_pool=None,
        scriptable=False,
        **kwargs
    )
    layers = dict([*model.named_modules()]).keys()
    # filter layers
    layers = [layer for layer in layers if "s3" in layer and "conv" in layer]
    print(layers)
    return model


def load_chresmax_abs_bypass_only(layername=None):
    kwargs = {
        'ip_scale_bands': 16,
        'classifier_input_size': 9216,
        'bypass': True,
        'c_debug': False,
    }
    model = create_model(
        'chresmax_abs_bypass_only',
        pretrained='/oscar/data/tserre/xyu110/pytorch-output/train/0/mnist/ip_16_chresmax_abs_bypass_only_gpu_8_cl_0.1_ip_3_224_224_9216_c1[_6,3,1_]_bypass',
        num_classes=10,
        in_chans=3,
        global_pool=None,
        scriptable=False,
        **kwargs
    )
    layers = dict([*model.named_modules()]).keys()
    # filter layers
    layers = [layer for layer in layers]
    print(layers)
    return model, 'chresmax_abs_bypass_only', layername, layers


def load_hmax_old_original():
    # checkpoint_path = "/oscar/data/tserre/npant1/pytorch-output/train/ip_18_hmax_old_gpu_1_cl_0.5_ip_3_224_224_0000_c1[_6,3,1_]_bypass_1/model_best.pth.tar"
    model = hmax_old_original()
    checkpoint_path = "/oscar/data/tserre/xyu110/hmax_image_scales/HMAX-epoch=59-val_acc1=99.36899038461539-val_loss=0.029037245774629693.ckpt"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Remove "HMAX." prefix from all keys in the state_dict
    state_dict = checkpoint['state_dict']
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('HMAX.'):
            new_key = key[5:]  # Remove "HMAX." prefix (5 characters)
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value
    
    # model = model.to(device).eval()
    model = model.to(device)
    model.load_state_dict(new_state_dict, strict=True)
    
    model.model_pre.base_scale = 224
    ip_scales = 18
    model.ip_scales = ip_scales
    model.scale = 2
    model.model_pre.ip_scales = ip_scales
    model.stream_2_bool = False
    
    
    
    layers = dict([*model.named_modules()]).keys()
    # filter layers
    layers = [layer for layer in layers]
    print(layers)
    
    return model, "hmax_old_original", None, None


def load_hmax_arjun():
    prj_name = "korean"
    n_ori = 4
    n_classes = 54 ## 54 characters
    lr = 1e-4
    weight_decay = 1e-4
    batch_size_per_gpu = 1
    num_epochs = 1 ## only 1 for few shot learning
    ip_scales = 18
    image_size = 224

    IP_bool = True
    IP_bool_recon = False
    IP_full_bool = False
    capsnet_bool = False
    IP_capsnet_bool = False
    IP_contrastive_bool = False
    lindeberg_fov_max_bool = False

    linderberg_bool = False
    my_data = True
    all_scales_train_bool = False
    orginal_mnist_bool = False

    oracle_bool = False
    argmax_bool = False

    oracle_plot_overlap_bool = False
    argmax_plot_overlap_bool = False
    oracle_argmax_plot_overlap_bool = False

    IP_bool = True
    IP_2_streams = False
    contrastive_2_bool = False
    sim_clr_bool = False

    IP_bool = False
    IP_2_streams = True
    ip_scales = 18

    # Mode
    test_mode = True
    val_mode = False
    continue_tr = False
    visualize_mode = False
    rdm_corr = False
    rdm_thomas = False
    featur_viz = False
    same_scale_viz = False
    cifar_data_bool = False

    scale_datasets = [18,36,8,24,30,12,4,20,16]
    train_dataset = 24

    MNIST_Scale = train_dataset
    
    # Initialize the model first
    model = hmax_fixed_ligtning.HMAX_trainer(prj_name, n_ori, 10, lr, weight_decay, ip_scales, IP_bool, visualize_mode, \
                                                    MNIST_Scale, capsnet_bool = capsnet_bool, IP_capsnet_bool = IP_capsnet_bool, \
                                                    IP_contrastive_bool = IP_contrastive_bool, lindeberg_fov_max_bool = lindeberg_fov_max_bool, \
                                                    IP_full_bool = IP_full_bool, IP_bool_recon = IP_bool_recon, IP_contrastive_finetune_bool = False, \
                                                    contrastive_2_bool = True, sim_clr_bool = True, batch_size = 32, \
                                                    IP_2_streams = IP_2_streams, cifar_data_bool = cifar_data_bool)
    
    # Load the model weights from regular PyTorch checkpoint
    checkpoint = torch.load('/oscar/data/tserre/xyu110/hmax_image_scales/HMAX-epoch=59-val_acc1=99.36899038461539-val_loss=0.029037245774629693.ckpt', map_location='cpu')
    
    # # Fix the key names to match the current model structure
    state_dict = checkpoint['state_dict']
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('model_pre.'):
            # Add 'HMAX.' prefix to match the current model structure
            new_key = 'HMAX.' + key
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value
    
    model.load_state_dict(new_state_dict)
    
    # Alternative PyTorch Lightning checkpoint loading (commented out)
    # model = hmax_fixed_ligtning.HMAX_trainer.load_from_checkpoint('./HMAX-epoch=59-val_acc1=99.36899038461539-val_loss=0.029037245774629693.ckpt')  
    
    model.HMAX.base_scale = image_size
    model.ip_scales = ip_scales
    model.HMAX.ip_scales = ip_scales
    model.HMAX.scale = 2

    if IP_2_streams:
        model.HMAX.model_pre.ip_scales = ip_scales
        model.HMAX.stream_2_bool = False

    return model, "hmax_old_arjun", None, None

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

def load_hmax_new_tricks(layername=None):
    kwargs = {
        'ip_scale_bands': 18,
        'classifier_input_size': 4096,
        'bypass': True,
        'c_debug': False,
    }
    # "/oscar/data/tserre/xyu110/pytorch-output/train/0/mnist/ip_18_hmax_old_gpu_1_cl_0.5_ip_3_224_224_0000_c1[_6,3,1_]_bypass_1/model_best.pth.tar",
    # /oscar/home/npant1/data/npant1/HMAX-epoch=59-val_acc1=99.36899038461539-val_loss=0.029037245774629693.ckpt
    model = create_model(
        'hmax_new_tricks',
        pretrained="/oscar/data/tserre/xyu110/pytorch-output/train/0/mnist/ip_18_hmax_new_tricks_gpu_8_cl_0.5_ip_3_224_224_0000_c1[_6,3,1_]_bypass_1/model_best.pth.tar",
        num_classes=10,
        in_chans=3,
        global_pool=None,
        scriptable=False,
        **kwargs
    )
    layers = dict([*model.named_modules()]).keys()
    # filter layers
    layers = [layer for layer in layers]
    print(layers)
    return model, "hmax_new_tricks", layername, layers


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
    layers = dict([*model.named_modules()]).keys()
    # filter layers
    layers = [layer for layer in layers]
    print(layers)
    return model, 'chresmax_v3_bypass_only', layername, layers

def load_models(modelname, layername=None):
    if modelname == 'hmax_old_original':
        # python korean.py --model_name hmax_old_original --layer_name model_pre.c2b
        return load_hmax_old_original()
    elif modelname == "hmax_old_arjun":
        # python korean.py --model_name hmax_old_arjun --layer_name HMAX.model_pre.c2b
        return load_hmax_arjun()
    elif modelname == 'hmax_old':
        return load_chmax(layername)
    elif modelname == 'chresmax_v3_bypass_only':
        return load_chresmax_v3_bypass_only(layername)
    elif modelname == 'chresmax_abs_bypass_only':
        return load_chresmax_abs_bypass_only(layername)
    elif modelname == 'hmax_new_tricks':
        return load_hmax_new_tricks(layername)
    else:
        raise ValueError(f"Unknown model name: {modelname}")

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Hangul character evaluation for a single model layer.")
    parser.add_argument('--model_name', type=str, default="hmax_old_original", choices=['hmax_old_arjun', 'hmax_old_original', 'hmax_old', 'chresmax_v3_bypass_only', 'chresmax_abs_bypass_only', 'hmax_new_tricks'], help='The name of the model to load.')
    parser.add_argument('--layer_name', type=str, default="model_pre.c2b", help='The specific layer to evaluate.')
    
    args = parser.parse_args()
    
    
    layer_to_process = args.layer_name
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # model, modelname, layername, all_layers = load_chmax("")
    # model, modelname, layername, all_layers = load_chresmax_v3_bypass_only("")
    # model, modelname, layername, all_layers = load_chresmax_abs_bypass_only("")
    # model, modelname, layername, all_layers = load_hmax_new_tricks("")
    
    model, modelname, _, _ = load_models(args.model_name)
    model = model.to(device)
    print(vars(model))
    print(f"Loaded model: {modelname} with layer: {layer_to_process}")
    try:
        korean = Korean(model,
                        os.path.join('/oscar/data/tserre/xyu110/pytorch-output/korean', modelname),
                        device,
                        '/gpfs/data/tserre/npant1/hangul_data',
                        224,
                        layer_to_process)
        korean.run()
    except Exception as e:
        print(f"Error running Korean experiment for layer {layer_to_process}: {e}")
        # write error to txt file
        with open(os.path.join(korean.outdir, 'error_log.txt'), 'a') as f:
            f.write(f"Error running Korean experiment for layer {layer_to_process}: {e}\n")
                

