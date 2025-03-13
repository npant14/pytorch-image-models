import sys
sys.path.insert(0, "/gpfs/data/tserre/npant1/brain_score")
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
import tqdm
import os
import numpy as np
import cv2
import time
import _pickle as pickle
from torch.utils.data import DataLoader, TensorDataset
import os
import pickle
import pandas as pd
from collections import OrderedDict

print(sys.path)

import brainscore
from brainscore.benchmarks.public_benchmarks import MajajHongITPublicBenchmark
import rsatoolbox

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class Brainscore_Experiment():
    def __init__(self, model, outdir, device='cpu'):
        self.model = model
        self.outdir = outdir
        self.device = device
        self.benchmark = MajajHongITPublicBenchmark()



    # '''
    # Expecting Input to be of the shape --> Catxnum_samplesxfeat_len
    # '''
    def rdm_corr_calc(self, small_state_features, large_state_features, c_stage):

        # Get the average feature for each category
        small_state_features_mean = np.mean(small_state_features, axis=1)   
        large_state_features_mean = np.mean(large_state_features, axis=1)   

        # Z Normalize seperately for each category
        eps = 1e-3
        small_state_features_z_norm = (small_state_features_mean - np.mean(small_state_features_mean, axis = 1, keepdims = True)) / (np.std(small_state_features_mean, axis = 1, keepdims = True) + eps)
        large_state_features_z_norm = (large_state_features_mean - np.mean(large_state_features_mean, axis = 1, keepdims = True)) / (np.std(large_state_features_mean, axis = 1, keepdims = True) + eps)

        # # Preparing data for calculating RDM
        small_state_features_data = rsatoolbox.data.Dataset(small_state_features_z_norm)
        large_state_features_data = rsatoolbox.data.Dataset(large_state_features_z_norm)

        # Build the 2 RDM Matrices by taking pairwise category euclidean distance for the 2 states
        # calc_rmd returns a RDMs object
        small_state_features_rdm = rsatoolbox.rdm.calc_rdm(small_state_features_data, method='euclidean', descriptor=None, noise=None)
        large_state_features_rdm = rsatoolbox.rdm.calc_rdm(large_state_features_data, method='euclidean', descriptor=None, noise=None)

        # Plotting and Saving
        # Need to write code for saving

        fig, ax, ret_val = rsatoolbox.vis.show_rdm(small_state_features_rdm, show_colorbar='figure')
        img_name = 'small_state_features_rdm_' + c_stage + '.png'
        save_path = os.path.join(self.outdir, img_name)
        fig.savefig(save_path, bbox_inches='tight', dpi=300)

        fig, ax, ret_val = rsatoolbox.vis.show_rdm(large_state_features_rdm, show_colorbar='figure')
        img_name = 'large_state_features_rdm_' + c_stage + '.png'
        save_path = os.path.join(self.outdir, img_name)
        fig.savefig(save_path, bbox_inches='tight', dpi=300)
        
        # Getting the SPearman Rank Correlation for the 2 RDMs
        spearman_corr = rsatoolbox.rdm.compare_spearman(small_state_features_rdm, large_state_features_rdm)

        print('spearman_corr : ', spearman_corr)

        return spearman_corr

    def save_tensor(self, filter_maps, scale, category, stage):
        filter_tensor = torch.concatenate(filter_maps, dim = 0)

        filter_numpy = filter_tensor.clone().cpu().numpy()    

        ############################################################################################

        # Option 4 --> FLatten
        filter_numpy = filter_numpy.reshape(filter_numpy.shape[0], -1)

        ############################################################################################

        job_dir = os.path.join(self.outdir, "rdm_corr")
        file_name = os.path.join(job_dir, f"filters_data_{scale}.pkl")

        open_file = open(file_name, "rb")
        try:
            filters_data = pickle.load(open_file)
        except EOFError:
            print(f"failed to open file {file_name}")
            filters_data = {}
        open_file.close()

        key_name = stage + '_scale_' + str(scale) + '_cat_' + str(category)

        if key_name in filters_data:
            filters_data[key_name] = np.concatenate([filters_data[key_name], filter_numpy], axis = 0)
        else:
            filters_data[key_name] = filter_numpy
        
        open_file = open(file_name, "wb")
        pickle.dump(filters_data, open_file)
        open_file.close()

#####################################################################################################
    def get_data_for_cat_scale(self, df, cat, minscale, maxscale, all_ids, all_Ys):
        cat_filter = df[df["category_name"] == cat]
        upper_cut = cat_filter[cat_filter['s'] <= maxscale]
        lower_cut = upper_cut[upper_cut['s'] >= minscale]
        ids = lower_cut['image_id'].to_numpy()

        ys = []
        relevant_ids = []

        for num, id in enumerate(ids):
            idxs = np.where(all_ids == id)
            if len(idxs) > 0:
                idx = np.where(all_ids == id)[0]

            if all_Ys[idx].shape[0] != 0: # it does not exist if this is the case
                ys.append(all_Ys[idx])
                relevant_ids.append(id)

        ys = np.vstack(ys)
        relevant_ids = np.array(relevant_ids)
        ## trimming for memory issues
        n = 100
        if relevant_ids.shape[0] > n:
            index = np.random.choice(relevant_ids.shape[0], n, replace=False)  
            relevant_ids = relevant_ids[index]
            ys = ys[index]
        return relevant_ids, ys

    def load_ids_as_images(self, ids):
        images_paths = [self.benchmark._assembly.stimulus_set.get_stimulus(img_id) for img_id in ids]
        images = np.array([cv2.imread(str(p)) for p in images_paths])
        images = images.astype(np.float64)
        images = torch.from_numpy(images)
        images = images.permute(0, 3, 1, 2)
        resize_transform = transforms.Resize((224, 224))

        images = torch.stack([resize_transform(img) for img in images]).numpy()
        return images

    def get_brainscore_dataloader(self, cat, s_i, df, all_ids, all_ys):
        ## just iterate through scale ranges
        ## load in for cat
        scales = [(0.75,0.85),(0.85,0.95),(1,1),(1.05,1.15),(1.15,1.25),(1.25,1.35)]
        filt_ids, filt_ys = self.get_data_for_cat_scale(df, cat, scales[int(s_i)][0], scales[int(s_i)][1], all_ids=all_ids, all_Ys=all_ys)
        filt_images = self.load_ids_as_images(filt_ids)
        tensor_x = torch.Tensor(filt_images) 
        tensor_y = torch.Tensor(filt_ys)
        dataset = TensorDataset(tensor_x,tensor_y) # create datset
        dataloader = DataLoader(dataset, batch_size=5) # create dataloader
        return dataloader

    def get_brainscore_data(self, cat, s_i, df, all_ids, all_ys):
        scales = [(0.75,0.85),(0.85,0.95),(1,1),(1.05,1.15),(1.15,1.25),(1.25,1.35)]
        filt_ids, filt_ys = self.get_data_for_cat_scale(df, cat, scales[int(s_i)][0], scales[int(s_i)][1], all_ids=all_ids, all_Ys=all_ys)
        filt_images = self.load_ids_as_images(filt_ids)
        tensor_x = torch.Tensor(filt_images) 
        tensor_y = torch.Tensor(filt_ys)
        return tensor_x, tensor_y

    def count_parameters(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def load_model(self, model):

        if torch.cuda.device_count() >= 1:
            print('We have multiple GPUs detected')
            model = model.to(device)
            model = nn.DataParallel(model)
        else:
            print('No GPU detected!')
            raise NotImplementedError()

        return model


    def accuracy(self, output, target, topk=(1,)):
        with torch.no_grad():
            _, pred = output.topk(max(topk), dim=1, largest=True, sorted=True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))
            res = [correct[:k].sum().item() for k in topk]
            return res

#####################################################################################################

    def rdm_corr_func(self, scale_test_list, save_rdms_list, num_samples = 1, recalculate_data=True):

        print("starting! ")
        assert len(scale_test_list) == 2, "For RDM Correlation only 2 scales should be given"

        # Paralelizing Model
        model = self.load_model(self.model)

        # Change Path into own folder
        # Scale 1
        job_dir = os.path.join(self.outdir, "rdm_corr")
        os.makedirs(job_dir, exist_ok=True)
        file_name = os.path.join(job_dir, f"filters_data_{scale_test_list[0]}.pkl")
        print(f"file name for scale 1: {file_name}")

        open_file = open(file_name, "wb")
        pickle.dump({'Empty1':0}, open_file)
        open_file.close()

        # Scale 2
        file_name = os.path.join(job_dir, f"filters_data_{scale_test_list[1]}.pkl")
        print(f"file name for scale 2: {file_name}")

        open_file = open(file_name, "wb")
        pickle.dump({'Empty2':0}, open_file)
        open_file.close()

        train_ids = np.load('/gpfs/data/tserre/npant1/brainscore/train_ids.npy', allow_pickle=True)
        Y_train = np.load('/gpfs/data/tserre/npant1/brainscore/y_train.npy')
        df = pd.read_csv("/gpfs/data/tserre/npant1/brainscore/image_dicarlo_hvm-public/image_dicarlo_hvm-public.csv")

        print(scale_test_list)
        if recalculate_data:
            for s_i, scale_bin in enumerate(scale_test_list):

                print('###################################################')
                print('This is scale : ',scale_bin)

                for c_i in ['Cars', 'Tables', 'Faces', 'Fruits', 'Planes', 'Boats', 'Animals', 'Chairs']:
                    torch.cuda.empty_cache()

                    print('###################################################')
                    print('This is category : ',c_i)

                    # SCALE_BIN 2 CORRESPONDS TO SCALE 1 
                    # SCALE_BINS [(0.75,0.85),(0.85,0.95),(1,1),(1.05,1.15),(1.15,1.25),(1.25,1.35)]
                    dload_test = self.get_brainscore_dataloader(c_i, scale_bin, df, train_ids, Y_train)
                    # Test Class
                    tester = Test(model, dload_test)

                    # create the hooks in the model
                    tester.feats = {}
                    hooks = []
                    for name, module in tester.model.named_modules():
                        # print(name)
                        hooks.append(module.register_forward_hook(tester.getActivation(name)))

                    # runs the model
                    records = tester()

                    # save the data so that you don't have to rerun every time
                    for layer in save_rdms_list:
                        self.save_tensor(tester.feats[layer], scale_bin, category=c_i, stage = layer)
                        del(tester.feats[layer])
                    tester.feats = {}

                    for hook in hooks:
                        hook.remove()

        # print('###################################################')
        # print('###################################################')
        # print('Now Loading the Data for sending to RDM Corr')

        # Change Path into own folder
        job_dir = os.path.join(self.outdir, "rdm_corr")
        file_name = os.path.join(job_dir, f"filters_data_{scale_test_list[0]}.pkl")

        open_file = open(file_name, "rb")
        filters_data_1 = pickle.load(open_file)
        print('filters_data : ',filters_data_1.keys())
        open_file.close()

        # Change Path into own folder
        file_name = os.path.join(job_dir, f"filters_data_{scale_test_list[1]}.pkl")

        open_file = open(file_name, "rb")
        filters_data_2 = pickle.load(open_file)
        print('filters_data : ',filters_data_2.keys())
        open_file.close()

        filters_data = {**filters_data_1, **filters_data_2}
        print(filters_data.keys())


        stage_list = save_rdms_list

        spearman_corr_dict = {}
        for stage in stage_list:
            avg_cor = []
            for _ in range(num_samples):
                small_scale = []
                large_scale = []
                for s_i, s_data in enumerate(scale_test_list):
                    for c_i in ['Cars', 'Tables', 'Faces', 'Fruits', 'Planes', 'Boats', 'Animals', 'Chairs']:
                        key_name = stage + '_scale_' + str(s_data) + '_cat_' + str(c_i)
                        temp_data = filters_data[key_name][:]

                        #if scale bin 2 (the x1 scale), split the data in half and sample from each half
                        if s_data == 2:
                            if s_i == 0:
                                random_indices = np.random.choice(temp_data.shape[0]//2, size=20, replace=False)
                            else:
                                random_indices = [x + temp_data.shape[0]//2 for x in np.random.choice(temp_data.shape[0]//2, size=20, replace=False)]

                        else:
                            random_indices = np.random.choice(temp_data.shape[0], size=20, replace=False)

                        temp_data = temp_data[random_indices]
                        if s_i == 0:
                            small_scale.append(temp_data)
                        else:
                            large_scale.append(temp_data)


                small_scale = np.stack(small_scale, axis=0)
                large_scale = np.stack(large_scale, axis=0)

                print('###################################################')
                print('Going to spearman_corr : ',stage)

                spearman_corr = self.rdm_corr_calc(small_scale, large_scale, stage)
                avg_cor.append(spearman_corr)

            spearman_corr_dict[stage] = np.mean(np.array(avg_cor))

        # Saving spearman dict as csv
        # Convert the dictionary to a DataFrame
        df = pd.DataFrame(list(spearman_corr_dict.items()), columns=["Scale", "SpearmanCorrelation"])
        # Save the DataFrame as a CSV file
        df.to_csv(os.path.join(job_dir, "spearman_corr.csv"), index=False)


        spearman_corr_list = list(spearman_corr_dict.values())
        print(f"results for {scale_test_list} : ")
        print(spearman_corr_dict)

        # Plot
        fig, ax = plt.subplots(1,1) 
        ax.scatter(range(1,len(spearman_corr_list)+1),spearman_corr_list)

        # Set number of ticks for x-axis
        ax.set_xticks(range(1,len(spearman_corr_list)+1))
        # Set ticks labels for x-axis
        ax.set_xticklabels(stage_list, rotation='vertical', fontsize=18)

        ax.set_xlabel("Pooling Stages", fontweight="bold")
        ax.set_ylabel("RDM Correlation", fontweight="bold")

        ax.set_ylim(0, 1)

        ax.grid()

        # Change Path into own folder
        fig.savefig(os.path.join(job_dir, "rdm_correlation_plot.png"), dpi=199)


class Test(object):
    #TODO: Figure out what can be deleted from here
    def __init__(self, model, data_loader):
        self.name = 'test'
        self.model = model
        self.data_loader = data_loader
        self.loss = nn.CrossEntropyLoss(size_average=False)
        self.loss = self.loss.to(device)
        self.feats = {}

    def getActivation(self,name):
        # the hook signature
        def hook(model, input, output):
            batch = output
            if type(batch) == list:
                batch = torch.stack(batch)
            if name in self.feats:
                self.feats[name].append(batch.detach())
            else:
                self.feats[name] = [batch.detach()]
        return hook   
###########################################################################

    def __call__(self):
        self.model.eval()

        start = time.time()
        record = {'loss': 0, 'top1': 0, 'top5': 0}
        with torch.no_grad():
            for (inp, target) in tqdm.tqdm(self.data_loader, desc=self.name):

                inp = inp.to(device)
                target = target.to(device)
                target = target.reshape(-1)
                output = self.model(inp)
        record['dur'] = (time.time() - start) / len(self.data_loader)

        return record


if __name__ == '__main__':

    print("running main")
    # Test Cases
    
    

