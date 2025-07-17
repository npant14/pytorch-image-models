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


from timm.models import create_model, load_checkpoint, is_model, list_models

from timm.models.HMAX_old import hmax_old_original


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


    def get_accuracy(self, filepaths):
        means = {}
        errs = {}
        maxes = {}
        # iterate through all the saved csvs
        for path in tqdm(filepaths, desc="Evaluating CSV Accuracy"):
            target_size, test_size = path.split("/")[-1].split(".")[0].split("-")
            # want to check both directions
            for transpose in [False, True]:
                CSVData = open(path)
                correlations = np.loadtxt(CSVData, delimiter=",")

                if transpose:
                    correlations = correlations.transpose()
                    target_size, test_size = test_size, target_size

                normalized = correlations
                normalized = (correlations - np.min(correlations, axis=0)) / (np.max(correlations) - np.min(correlations))

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
    
    def set_layer(self, layer_name):
        self.layer = layer_name

    def run(self):
        pairs = [(13, 13), (13, 52), (13, 130)]
        filepaths = [os.path.join(self.outdir, f"{size_1}-{size_2}.csv") for (size_1, size_2) in pairs]
        self.create_correlation_matrices(pairs, self.layer)
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


def load_old_hmax():
    checkpoint_path = "/oscar/data/tserre/npant1/pytorch-output/train/ip_18_hmax_old_gpu_1_cl_0.5_ip_3_224_224_0000_c1[_6,3,1_]_bypass_1/model_best.pth.tar"
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = hmax_old_original().to(device).eval()
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    model.model_pre.base_scale = 224
    ip_scales = 18
    model.ip_scales = ip_scales
    model.scale = 2
    model.model_pre.ip_scales = ip_scales
    model.stream_2_bool = False
    
    # print(vars(model))
    
    layers = dict([*model.named_modules()]).keys()
    # filter layers
    layers = [layer for layer in layers]
    print(layers)
    
    return model, "hmax_old_original", None, None


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
    layers = dict([*model.named_modules()]).keys()
    # filter layers
    layers = [layer for layer in layers]
    print(layers)
    return model, 'chresmax_v3_bypass_only', layername, layers

def load_models(modelname, layername=None):
    if modelname == 'hmax_old_original':
        return load_old_hmax()
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


def test_loaded_model(model):
    if next(model.parameters()).is_cuda:
        print("✅ Model is loaded to GPU.")
    else:
        print("❌ Model is NOT on GPU.")

    try:
        dummy_input = torch.randn(1, 3, 224, 224).to(device)
        model.eval()
        with torch.no_grad():
            output = model(dummy_input)
        print("✅ Model forward pass successful.")
    except Exception as e:
        print(f"❌ Model forward pass failed: {e}")
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Hangul character evaluation for a single model layer.")
    parser.add_argument('--model_name', type=str, choices=['hmax_old_original', 'hmax_old', 'chresmax_v3_bypass_only', 'chresmax_abs_bypass_only', 'hmax_new_tricks'], help='The name of the model to load.')
    parser.add_argument('--layer_name', type=str, help='The specific layer to evaluate.')
    
    args = parser.parse_args()
    
    
    layer_to_process = args.layer_name
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # model, modelname, layername, all_layers = load_chmax("")
    # model, modelname, layername, all_layers = load_chresmax_v3_bypass_only("")
    # model, modelname, layername, all_layers = load_chresmax_abs_bypass_only("")
    # model, modelname, layername, all_layers = load_hmax_new_tricks("")
    
    model, modelname, _, _ = load_models(args.model_name)
    
    test_loaded_model(model)

    model = model.to(device)
    
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
                

