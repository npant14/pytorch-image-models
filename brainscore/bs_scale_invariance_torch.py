import numpy as np
import pandas as pd
import torch
import timm
import cv2
import os
from sklearn.cross_decomposition import PLSRegression
from torchvision import transforms
from brainscore.benchmarks.public_benchmarks import MajajHongITPublicBenchmark
from PIL import Image
from constants import models_meta

meta={}
for s in models_meta :
  if s['fields']['model'] not in meta:
    meta[s['fields']['model']]={s['fields']['key']:s['fields']['value']}
  else:
    meta[s['fields']['model']][s['fields']['key']]=s['fields']['value']


# Convert meta to easy-lookup dictionary
layer_config = {}
for entry in models_meta:
    model_name = entry['fields']['model']
    layer_type = entry['fields']['key']
    layer_value = entry['fields']['value']
    if model_name not in layer_config:
        layer_config[model_name] = {}
    layer_config[model_name][layer_type] = layer_value

benchmark = MajajHongITPublicBenchmark()
df = benchmark._assembly.stimulus_set

train_ids = np.load('/gpfs/data/tserre/npant1/brainscore/train_ids.npy', allow_pickle=True)
test_ids = np.load('/gpfs/data/tserre/npant1/brainscore/test_ids.npy', allow_pickle=True)
all_ids = np.concatenate((train_ids,test_ids))

Y_train = np.load('/gpfs/data/tserre/npant1/brainscore/y_train.npy')
Y_test = np.load('/gpfs/data/tserre/npant1/brainscore/y_test.npy')
Y_all = np.concatenate((Y_train,Y_test))




SCALES = [(0.75, 0.85), (0.85, 0.95), (1, 1), (1.05, 1.15), (1.15, 1.25), (1.25, 1.35)]
RXY_RANGES = [
    (-46.43, -27.38),
    (-27.379, -12.01),
    (-12.009, -0.001),
    (-0.009, 0.03),
    (0.021, 19.56),
    (19.561, 52.17)
]

RXZ_RANGES = [
    (-44.94, -29.42),
    (-29.419, -14.77),
    (-14.769, -0.021),
    (-0.009, 0.02),
    (0.011, 22.69),
    (22.691, 44.94)
]

RYZ_RANGES = [
    (-93.34, -30.24),
    (-30.239, -14.66),
    (-14.659, -0.001),
    (-0.009, 0.41),
    (0.401, 23.88),
    (23.881, 93.39)
]

TZ_RANGES = [
    (-0.601, -0.39),
    (-0.389, -0.211),
    (-0.21, -0.0011),
    (-0.001, 0.099),
    (0.1, 0.249),
    (0.25, 0.601)
]

TY_RANGES = [
    (-0.301, -0.202),
    (-0.201, -0.102),
    (-0.101, -0.0011),
    (-0.001, 0.099),
    (0.1, 0.199),
    (0.2, 0.301)
]

REF_SCALE_IDX = 2
RXY_REF_IDX = 3
RXZ_REF_IDX = 3
RYZ_REF_IDX = 3
TZ_REF_IDX = 3
TY_REF_IDX = 3

BATCH_SIZE = 64
results_csv = "../results/model_scores_v5.csv"

if os.path.exists(results_csv):
    results_df = pd.read_csv(results_csv)
else:
    results_df = pd.DataFrame(columns=['model', 'layer', 'train_scale', 'test_scale', 'score'])

def get_data_trans(df, mint, maxt, all_ids, all_Ys):
    """
    Filters the dataframe for a given translation range (irrespective of category)
    and returns the corresponding image ids and labels.
    """
   
    scale_filtered = df[(df['tz'] >= mint) & (df['tz'] <= maxt)]
    ids = scale_filtered['image_id'].to_numpy()

    ys = []
    relevant_ids = []
    for id in ids:
        idx = np.where(all_ids == id)[0]
        if idx.size > 0 and all_Ys[idx].shape[0] != 0:
            ys.append(all_Ys[idx])
            relevant_ids.append(id)
    ys = np.vstack(ys)
    relevant_ids = np.array(relevant_ids)
    return relevant_ids, ys

def get_data_rot(df, minr, maxr, all_ids, all_Ys):
    """
    Filters the dataframe for a given scale range (irrespective of category)
    and returns the corresponding image ids and labels.
    """
    scale_filtered = df[(df['rxy'] >= minr) & (df['rxy'] <= maxr)]
    ids = scale_filtered['image_id'].to_numpy()
    
    ys = []
    relevant_ids = []
    for id in ids:
        idx = np.where(all_ids == id)[0]
        if idx.size > 0 and all_Ys[idx].shape[0] != 0:
            ys.append(all_Ys[idx])
            relevant_ids.append(id)
    ys = np.vstack(ys)
    relevant_ids = np.array(relevant_ids)
    return relevant_ids, ys

def get_data_scale(df, minscale, maxscale, all_ids, all_Ys):
    """
    Filters the dataframe for a given scale range (irrespective of category)
    and returns the corresponding image ids and labels.

    Args:
        df (pd.DataFrame): DataFrame with image metadata.
        minscale (float): Minimum scale value.
        maxscale (float): Maximum scale value.
        all_ids (np.ndarray): Array of image ids.
        all_Ys (np.ndarray): Array of labels/features.

    Returns:
        relevant_ids (np.ndarray): Array of filtered image ids.
        ys (np.ndarray): Associated labels/features.
    """
    
    scale_filtered = df[(df['s'] >= minscale) & (df['s'] <= maxscale)]
    ids = scale_filtered['image_id'].to_numpy()

    ys = []
    relevant_ids = []
    for id in ids:
        idx = np.where(all_ids == id)[0]
        if idx.size > 0 and all_Ys[idx].shape[0] != 0:
            ys.append(all_Ys[idx])
            relevant_ids.append(id)
    
    ys = np.vstack(ys)
    relevant_ids = np.array(relevant_ids)
    return relevant_ids, ys


def load_ids_as_images(ids, input_size,preprocessing):
    images_paths = [benchmark._assembly.stimulus_set.get_image(img_id) for img_id in ids]
    # loading images using PIL
    images = [preprocessing(Image.open(p).resize(input_size)) for p in images_paths]
    #images = [cv2.resize(cv2.imread(str(p)), input_size) for p in images_paths]
    images = np.stack(images).astype(np.float32)
    #images = images.transpose(0, 3, 1, 2)
    return torch.tensor(images)


def get_activations(model, images, layer_name, batch_size=32):
    activations = []

    def hook(module, input, output):
        activations.append(output.detach().cpu().numpy())

    handle = dict(model.named_modules())[layer_name].register_forward_hook(hook)
    activations = []
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            model(batch)
    handle.remove()

    return np.concatenate(activations)


def brain_score(X_train, Y_train, X_test, Y_test):
    pls = PLSRegression(n_components=25, scale=False)
    
    pls_kernel = pls.fit(X_train, Y_train).coef_
    Y_pred = np.dot(X_test, pls_kernel)
    correlations = [np.corrcoef(Y_test[:, i], Y_pred[:, i])[0, 1] for i in range(Y_test.shape[1])]
    return np.median(correlations)


def process_and_score(model, X_train, X_test, Y_train, Y_test, layer_name, batch_size=32):
    """
    Process images through model and compute brain score.
    
    Args:
        model: The neural network model
        X_train: Training images
        X_test: Test images
        Y_train: Training labels
        Y_test: Test labels
        layer_name: Name of the layer to extract activations from
        batch_size: Batch size for processing
        
    Returns:
        float: Brain score
    """
    # Get activations
    X_train_act = get_activations(model, X_train, layer_name)
    X_test_act = get_activations(model, X_test, layer_name)
    
    # Compute score
    score = brain_score(X_train_act, Y_train, X_test_act, Y_test)
    
    # Clear GPU memory
    del X_train_act, X_test_act
    torch.cuda.empty_cache()
    
    return score

from scipy.stats import pearsonr

def calculate_ceiling(Y_train,Y_test):
    Y_train = Y_train.astype(np.float32)
    Y_test = Y_test.astype(np.float32)
   
    r, _ = pearsonr(Y_test.flatten(), Y_train.flatten())
   
    if (1 + r) != 0:
        reliability = (2 * r) / (1 + r)
    else:
        reliability = np.nan

    return reliability



train_scale_ids, Y_train_scale = get_data_scale(df, *SCALES[2], all_ids, Y_all)
train_rot_ids_rxz, Y_train_rot_rxz = get_data_rot(df, *RXZ_RANGES[RXZ_REF_IDX], all_ids, Y_all)
train_rot_ids_ryz, Y_train_rot_ryz = get_data_rot(df, *RYZ_RANGES[RYZ_REF_IDX], all_ids, Y_all)
train_rot_ids_rxy, Y_train_rot_rxy = get_data_rot(df, *RXY_RANGES[RXY_REF_IDX], all_ids, Y_all)

train_trans_ids_tz, Y_train_trans_tz = get_data_trans(df, *TZ_RANGES[TZ_REF_IDX], all_ids, Y_all)
train_trans_ids_ty, Y_train_trans_ty = get_data_trans(df, *TY_RANGES[TY_REF_IDX], all_ids, Y_all)

saving_data =[]

for ix, scale in enumerate(SCALES):
    rot_rxz = RXZ_RANGES[ix]
    rot_ryz = RYZ_RANGES[ix]
    rot_rxy = RXY_RANGES[ix]
    trans_tz = TZ_RANGES[ix]
    trans_ty = TY_RANGES[ix]
    
    # Scale
    test_scale_ids, Y_test_scale = get_data_scale(df, *scale, all_ids, Y_all)
    # Get reference scores if:
    test_rot_rxz_ids, Y_test_rot_rxz = get_data_rot(df, *rot_rxz, all_ids, Y_all)
    test_rot_ryz_ids, Y_test_rot_ryz = get_data_rot(df, *rot_ryz, all_ids, Y_all)
    test_rot_rxy_ids, Y_test_rot_rxy = get_data_rot(df, *rot_rxy, all_ids, Y_all)
    test_trans_tz_ids, Y_test_trans_tz = get_data_trans(df, *trans_tz, all_ids, Y_all)
    test_trans_ty_ids, Y_test_trans_ty = get_data_trans(df, *trans_ty, all_ids, Y_all)

    print('scale:',scale)
    size = min(len(Y_train_scale), len(Y_test_scale))
    Y_train_scale = Y_train_scale[:size,:]
    Y_test_scale = Y_test_scale[:size,:]
    r_scale = brain_score(Y_train_scale,Y_train_scale,Y_test_scale,Y_test_scale)
    print('r_scale:',r_scale)


    print('rot_rxz')
    size_rot = min(len(Y_train_rot_rxz), len(Y_test_rot_rxz))
    Y_train_rot_rxz = Y_train_rot_rxz[:size_rot,:]
    Y_test_rot_rxz = Y_test_rot_rxz[:size_rot,:]
    #print(Y_train_rot_rxz.shape, Y_test_rot_rxz.shape)
    r_rot_rxz = brain_score(Y_train_rot_rxz,Y_train_rot_rxz,Y_test_rot_rxz,Y_test_rot_rxz)
    print('r_rot_rxz:',r_rot_rxz)


    print('rot_ryz')
    size_rot = min(len(Y_train_rot_ryz), len(Y_test_rot_ryz))
    Y_train_rot_ryz = Y_train_rot_ryz[:size_rot,:]
    Y_test_rot_ryz = Y_test_rot_ryz[:size_rot,:]
    print(Y_train_rot_ryz.shape, Y_test_rot_ryz.shape)  

    r_rot_ryz = brain_score(Y_train_rot_ryz,Y_train_rot_ryz,Y_test_rot_ryz,Y_test_rot_ryz)
    print('r_rot_ryz:',r_rot_ryz)

    print('rot_rxy')
    size_rot = min(len(Y_train_rot_rxy), len(Y_test_rot_rxy))
    Y_train_rot_rxy = Y_train_rot_rxy[:size_rot,:]
    Y_test_rot_rxy = Y_test_rot_rxy[:size_rot,:]
    print(Y_train_rot_rxy.shape, Y_test_rot_rxy.shape)

    r_rot_rxy = brain_score(Y_train_rot_rxy,Y_train_rot_rxy,Y_test_rot_rxy,Y_test_rot_rxy)
    print('r_rot_rxy:',r_rot_rxy)

    print('trans_tz')
    size_trans = min(len(Y_train_trans_tz),len(Y_test_trans_tz))
    Y_train_trans_tz = Y_train_trans_tz[:size_trans,:]
    Y_test_trans_tz = Y_train_trans_tz[:size_trans,:]
    r_trans_tz = brain_score(Y_train_trans_tz,Y_train_trans_tz,Y_test_trans_tz,Y_test_trans_tz)
    print('r_trans_tz:',r_trans_tz)
    
    saving_data.append([ix,scale,r_scale,rot_rxy,r_rot_rxy,rot_rxz,r_rot_rxz,rot_rxy,r_rot_ryz,trans_tz,r_trans_tz])  
    

saving_data = pd.DataFrame(saving_data)
saving_data.to_csv('referencedata.csv')

exit()

timm_models = timm.list_models(pretrained=True)




for num, model_name in enumerate(timm_models):
    if model_name in results_df['model'].values:
        continue
    if num%2 ==1: 
      import torch
      # Get current default folder
      try:
        print(torch.hub.get_dir())
        # clean up the hub directory
        import shutil
        shutil.rmtree(torch.hub.get_dir())
      except Exception as e:
        print(f"Error cleaning up hub directory: {e}")
      
    try:
        model = timm.create_model(model_name, pretrained=True).eval()
    except Exception as e:
        print(f"Error loading model {model_name}: {e}")
        continue
    config = timm.data.resolve_data_config({}, model=model)
    preprocessing = timm.data.create_transform(**config)
    input_size = config['input_size'][1:3]

    train_scale_ids, Y_train_scale = get_data_scale(df, *SCALES[2], train_ids, Y_train)
    train_rot_ids_rxz, Y_train_rot_rxz = get_data_rot(df, *RXZ_RANGES[RXZ_REF_IDX], train_ids, Y_train)
    train_rot_ids_ryz, Y_train_rot_ryz = get_data_rot(df, *RYZ_RANGES[RYZ_REF_IDX], train_ids, Y_train)
    train_rot_ids_rxy, Y_train_rot_rxy = get_data_rot(df, *RXY_RANGES[RXY_REF_IDX], train_ids, Y_train)
    
    train_trans_ids_tz, Y_train_trans_tz = get_data_trans(df, *TZ_RANGES[TZ_REF_IDX], train_ids, Y_train)
    train_trans_ids_ty, Y_train_trans_ty = get_data_trans(df, *TY_RANGES[TY_REF_IDX], train_ids, Y_train)
    
    X_train_scale = load_ids_as_images(train_scale_ids, input_size,preprocessing)
    X_train_rot_rxz = load_ids_as_images(train_rot_ids_rxz, input_size,preprocessing)
    X_train_rot_ryz = load_ids_as_images(train_rot_ids_ryz, input_size,preprocessing)
    X_train_rot_rxy = load_ids_as_images(train_rot_ids_rxy, input_size,preprocessing)   
    
    X_train_trans_tz = load_ids_as_images(train_trans_ids_tz, input_size,preprocessing)
    X_train_trans_ty = load_ids_as_images(train_trans_ids_ty, input_size,preprocessing)
    
    layers_to_test = list(dict(model.named_modules()).keys())[-4:]

    for layer_name in layers_to_test:
        for ix, scale in enumerate(SCALES):
            rot_rxz = RXZ_RANGES[ix]
            rot_ryz = RYZ_RANGES[ix]
            rot_rxy = RXY_RANGES[ix]
            trans_tz = TZ_RANGES[ix]
            trans_ty = TY_RANGES[ix]
            
            # Scale
            test_scale_ids, Y_test_scale = get_data_scale(df, *scale, test_ids, Y_test)
            X_test_scale = load_ids_as_images(test_scale_ids, input_size, preprocessing)
            score_scale = process_and_score(model, X_train_scale, X_test_scale, Y_train_scale, Y_test_scale, layers_to_test[-1])
            del X_test_scale
            
            # Rotation RXZ
            test_rot_rxz_ids, Y_test_rot_rxz = get_data_rot(df, *rot_rxz, test_ids, Y_test)
            test_rot_ryz_ids, Y_test_rot_ryz = get_data_rot(df, *rot_ryz, test_ids, Y_test)
            test_rot_rxy_ids, Y_test_rot_rxy = get_data_rot(df, *rot_rxy, test_ids, Y_test)
            
            test_trans_tz_ids, Y_test_trans_tz = get_data_trans(df, *trans_tz, test_ids, Y_test)
            test_trans_ty_ids, Y_test_trans_ty = get_data_trans(df, *trans_ty, test_ids, Y_test)

            X_test_rot_rxz = load_ids_as_images(test_rot_rxz_ids, input_size, preprocessing)
            X_test_rot_ryz = load_ids_as_images(test_rot_ryz_ids, input_size, preprocessing)
            X_test_rot_rxy = load_ids_as_images(test_rot_rxy_ids, input_size, preprocessing)
            X_test_trans_tz = load_ids_as_images(test_trans_tz_ids, input_size, preprocessing)
            X_test_trans_ty = load_ids_as_images(test_trans_ty_ids, input_size, preprocessing)
            
            # Process and score each transformation
            score_rot_rxz = process_and_score(model, X_train_rot_rxz, X_test_rot_rxz, Y_train_rot_rxz, Y_test_rot_rxz, layers_to_test[-1])
            score_rot_ryz = process_and_score(model, X_train_rot_ryz, X_test_rot_ryz, Y_train_rot_ryz, Y_test_rot_ryz, layers_to_test[-1])
            score_rot_rxy = process_and_score(model, X_train_rot_rxy, X_test_rot_rxy, Y_train_rot_rxy, Y_test_rot_rxy, layers_to_test[-1])
            score_trans_tz = process_and_score(model, X_train_trans_tz, X_test_trans_tz, Y_train_trans_tz, Y_test_trans_tz, layers_to_test[-1])
            score_trans_ty = process_and_score(model, X_train_trans_ty, X_test_trans_ty, Y_train_trans_ty, Y_test_trans_ty, layers_to_test[-1])
            
            # Clear GPU memory
            del X_test_rot_rxz, X_test_rot_ryz, X_test_rot_rxy, X_test_trans_tz, X_test_trans_ty

            result_row = pd.DataFrame([{
                'model': model_name,
                'layer': layer_name,
                'train_scale': str(SCALES[2]),
                'train_rot_rxz': str(RXZ_RANGES[RXZ_REF_IDX]),
                'train_rot_ryz': str(RYZ_RANGES[RYZ_REF_IDX]),
                'train_rot_rxy': str(RXY_RANGES[RXY_REF_IDX]),
                'train_trans_tz': str(TZ_RANGES[TZ_REF_IDX]),
                'train_trans_ty': str(TY_RANGES[TY_REF_IDX]),
                'test_rot_rxz': str(rot_rxz),
                'test_rot_ryz': str(rot_ryz),
                'test_rot_rxy': str(rot_rxy),
                'test_trans_tz': str(trans_tz),
                'test_trans_ty': str(trans_ty),
                'test_scale': str(scale),
                'scale_score': score_scale,
                'rot_rxz_score': score_rot_rxz,
                'rot_ryz_score': score_rot_ryz,
                'rot_rxy_score': score_rot_rxy,
                'trans_tz_score': score_trans_tz,
                'trans_ty_score': score_trans_ty
            }])

            results_df = pd.concat([results_df, result_row], ignore_index=True)
            results_df.to_csv(results_csv[:-4] + f"2.csv", index=False)
            print(f"Model {model_name}, Layer {layer_name}, Scale {scale}, Score: {score_scale}")