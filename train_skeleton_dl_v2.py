#!/usr/bin/env python3
"""
Single-model training script with optional scale augmentation & scale logits.

This code is adapted from your teacher-student script:
 - All teacher references have been removed.
 - We now train a single model that outputs (class_logits, scale_logits).
 - If your model doesn't return scale_logits, remove references to scale_logits.

Command-line arguments remain largely the same, except references to teacher/student have been removed or repurposed.

Author: ChatGPT
"""

import argparse
import importlib
import json
import logging
import os
import sys
import time
import random
import json
from collections import OrderedDict
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils
import yaml
from torch.nn.parallel import DistributedDataParallel as NativeDDP
import torch
try:
    torch.multiprocessing.set_start_method('spawn')
except:
    pass

try:
    from apex import amp
    from apex.parallel import DistributedDataParallel as ApexDDP
    from apex.parallel import convert_syncbn_model
    has_apex = True
except ImportError:
    has_apex = False
from timm import utils
from timm.data import create_dataset, create_loader, resolve_data_config, AugMixDataset, create_loader_scale
from timm.layers import convert_sync_batchnorm, set_fast_norm
from timm.models import create_model, safe_model_name, resume_checkpoint, model_parameters
from timm.optim import create_optimizer_v2, optimizer_kwargs
from timm.scheduler import create_scheduler_v2, scheduler_kwargs
from timm.utils import ApexScaler, NativeScaler
import matplotlib.pyplot as plt
has_native_amp = False
try:
    if getattr(torch.cuda.amp, 'autocast') is not None:
        has_native_amp = True
except AttributeError:
    pass

try:
    import wandb
    has_wandb = True
except ImportError:
    has_wandb = False

try:
    from functorch.compile import memory_efficient_fusion
    has_functorch = True
except ImportError as e:
    has_functorch = False

CSV_FILE= "/files22_lrsresearch/CLPS_Serre_Lab/projects/prj_concept_surgery/finetuning_models/fp_checked2.csv"
ROOT_DIR = "/oscar/data/tserre/npant1/ILSVRC/train"
MASK_LOOKUP_JSON = "/files22_lrsresearch/CLPS_Serre_Lab/projects/prj_hmax_masks/HMAX/SAM_Imagenet/sam2/image_to_mask_lookup.json"

if not os.path.exists(CSV_FILE):
    CSV_FILE = "/files22_lrsresearch/CLPS_Serre_Lab/projects/prj_concept_surgery/finetuning_models/fp_checked2.csv"
if not os.path.exists(MASK_LOOKUP_JSON):
    MASK_LOOKUP_JSON = '/users/irodri15/data/irodri15/Hmax/pytorch-image-models/timm/data/_info/image_to_mask_lookup.json'
import pandas as pd

csv_df = pd.read_csv(CSV_FILE)

lookup_file_scale = dict(zip(csv_df['Image File'], csv_df['scale_band']))


torch.autograd.set_detect_anomaly(True)
has_compile = hasattr(torch, 'compile')
_logger = logging.getLogger('train')

# First arg parser: for the config file.
config_parser = argparse.ArgumentParser(description='Training Config', add_help=False)
config_parser.add_argument('-c', '--config', default='', type=str, metavar='FILE',
                           help='YAML config file specifying default arguments')

parser = argparse.ArgumentParser(description='PyTorch Single-Model w/ Scale Logits Training')

# ----------------------------------------------------------------------
# Most of your original arguments are preserved, but references to teacher/student have been removed.
# ----------------------------------------------------------------------

# Dataset parameters
group = parser.add_argument_group('Dataset parameters')
parser.add_argument('data', nargs='?', metavar='DIR', const=None,
                    help='path to dataset (positional is *deprecated*, use --data-dir)')
parser.add_argument('--data-dir', metavar='DIR',
                    help='path to dataset (root dir)')
parser.add_argument('--dataset', metavar='NAME', default='',
                    help='dataset type + name ("<type>/<name>") (default: ImageFolder or ImageTar if empty)')
group.add_argument('--train-split', metavar='NAME', default='train',
                   help='dataset train split (default: train)')
group.add_argument('--val-split', metavar='NAME', default='validation',
                   help='dataset validation split (default: validation)')
parser.add_argument('--train-num-samples', default=None, type=int,
                    metavar='N', help='Manually specify num samples in train split, for IterableDatasets.')
parser.add_argument('--val-num-samples', default=None, type=int,
                    metavar='N', help='Manually specify num samples in validation split, for IterableDatasets.')
group.add_argument('--dataset-download', action='store_true', default=False,
                   help='Allow download of dataset for torch/ and tfds/ datasets that support it.')
group.add_argument('--class-map', default='', type=str, metavar='FILENAME',
                   help='path to class to idx mapping file (default: "")')
group.add_argument('--input-img-mode', default=None, type=str,
                   help='Dataset image conversion mode for input images.')
group.add_argument('--input-key', default=None, type=str,
                   help='Dataset key for input images.')
group.add_argument('--target-key', default=None, type=str,
                   help='Dataset key for target labels.')

# Model parameters
group = parser.add_argument_group('Model parameters')
group.add_argument('--model', default='resnet50', type=str, metavar='MODEL',
                   help='Name of model to train (default: "resnet50")')
group.add_argument('--pretrained', action='store_true', default=False,
                   help='Start with pretrained version of specified network (if avail)')
group.add_argument('--pretrained-path', default=None, type=str,
                   help='Load this checkpoint as if it were the pretrained weights (with adaptation).')
group.add_argument('--initial-checkpoint', default='', type=str, metavar='PATH',
                   help='Load this checkpoint into model after initialization (default: none)')
group.add_argument('--resume', default='', type=str, metavar='PATH',
                   help='Resume full model and optimizer state from checkpoint (default: none)')
group.add_argument('--no-resume-opt', action='store_true', default=False,
                   help='prevent resume of optimizer state when resuming model')
group.add_argument('--num-classes', type=int, default=None, metavar='N',
                   help='number of label classes (Model default if None)')
group.add_argument('--gp', default=None, type=str, metavar='POOL',
                   help='Global pool type, one of (fast, avg, max, avgmax, avgmaxc). Model default if None.')
group.add_argument('--img-size', type=int, default=None, metavar='N',
                   help='Image size (default: None => model default)')
group.add_argument('--in-chans', type=int, default=None, metavar='N',
                   help='Image input channels (default: None => 3)')
group.add_argument('--input-size', default=None, nargs=3, type=int,
                   metavar='N N N',
                   help='Input all image dimensions (d h w, e.g. --input-size 3 224 224), uses model default if empty')
group.add_argument('--crop-pct', default=None, type=float,
                   metavar='N', help='Input image center crop percent (for validation only)')
group.add_argument('--mean', type=float, nargs='+', default=None, metavar='MEAN',
                   help='Override mean pixel value of dataset')
group.add_argument('--std', type=float, nargs='+', default=None, metavar='STD',
                   help='Override std deviation of dataset')
group.add_argument('--interpolation', default='', type=str, metavar='NAME',
                   help='Image resize interpolation type (overrides model)')
group.add_argument('-b', '--batch-size', type=int, default=128, metavar='N',
                   help='Input batch size for training (default: 128)')
group.add_argument('-vb', '--validation-batch-size', type=int, default=None, metavar='N',
                   help='Validation batch size override (default: None)')
group.add_argument('--channels-last', action='store_true', default=False,
                   help='Use channels_last memory layout')
group.add_argument('--fuser', default='', type=str,
                   help="Select jit fuser. One of ('', 'te', 'old', 'nvfuser')")
group.add_argument('--grad-accum-steps', type=int, default=1, metavar='N',
                   help='The number of steps to accumulate gradients (default: 1)')
group.add_argument('--grad-checkpointing', action='store_true', default=False,
                   help='Enable gradient checkpointing through model blocks/stages')
group.add_argument('--fast-norm', default=False, action='store_true',
                   help='enable experimental fast-norm')
group.add_argument('--model-kwargs', nargs='*', default={}, action=utils.ParseKwargs)

# Single-model scale argument
parser.add_argument('--scale-bands', type=int, default=4,
                   help='Integer controlling the range of scaling factors for random_rescale.')
group.add_argument('--cl-lambda', default=0,  type=float,
                   help='lambda to scale cl term')
group.add_argument('--alpha', default=0.01,  type=float,
                   help='lambda to scale cl term')
# Device & distributed
group = parser.add_argument_group('Device parameters')
group.add_argument('--device', default='cuda', type=str,
                    help="Device (accelerator) to use.")
group.add_argument('--amp', action='store_true', default=False,
                   help='use NVIDIA Apex AMP or Native AMP for mixed precision training')
group.add_argument('--amp-dtype', default='float16', type=str,
                   help='lower precision AMP dtype (default: float16)')
group.add_argument('--amp-impl', default='native', type=str,
                   help='AMP impl to use, "native" or "apex" (default: native)')
group.add_argument('--no-ddp-bb', action='store_true', default=False,
                   help='Force broadcast buffers for native DDP to off.')
group.add_argument('--synchronize-step', action='store_true', default=False,
                   help='torch.cuda.synchronize() end of each step')
group.add_argument("--local_rank", default=0, type=int)
parser.add_argument('--device-modules', default=None, type=str, nargs='+',
                    help="Python imports for device backend modules.")

# Optimizer parameters
group = parser.add_argument_group('Optimizer parameters')
group.add_argument('--opt', default='sgd', type=str, metavar='OPTIMIZER',
                   help='Optimizer (default: "sgd")')
group.add_argument('--opt-eps', default=None, type=float, metavar='EPSILON',
                   help='Optimizer Epsilon (default: None, use opt default)')
group.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                   help='Optimizer Betas (default: None, use opt default)')
group.add_argument('--momentum', type=float, default=0.9, metavar='M',
                   help='Optimizer momentum (default: 0.9)')
group.add_argument('--weight-decay', type=float, default=2e-5,
                   help='weight decay (default: 2e-5)')
group.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                   help='Clip gradient norm (default: None, no clipping)')
group.add_argument('--clip-mode', type=str, default='norm',
                   help='Gradient clipping mode. One of ("norm", "value", "agc")')
group.add_argument('--layer-decay', type=float, default=None,
                   help='layer-wise learning rate decay (default: None)')
group.add_argument('--opt-kwargs', nargs='*', default={}, action=utils.ParseKwargs)

# Learning rate schedule parameters
group = parser.add_argument_group('Learning rate schedule parameters')
group.add_argument('--sched', type=str, default='cosine', metavar='SCHEDULER',
                   help='LR scheduler (default: "step"')
group.add_argument('--lr', type=float, default=None, metavar='LR',
                   help='learning rate, overrides lr-base if set (default: None)')
group.add_argument('--lr-cycle-decay', type=float, default=0.5, metavar='MULT',
                   help='amount to decay each learning rate cycle (default: 0.5)')
group.add_argument('--warmup-lr', type=float, default=1e-5, metavar='LR',
                   help='warmup learning rate (default: 1e-5)')
group.add_argument('--epochs', type=int, default=300, metavar='N',
                   help='number of epochs to train (default: 300)')
group.add_argument('--epoch-repeats', type=float, default=0., metavar='N',
                   help='epoch repeat multiplier (number of times to repeat dataset epoch per train epoch).')
group.add_argument('--decay-epochs', type=float, default=90, metavar='N',
                   help='epoch interval to decay LR')
group.add_argument('--warmup-epochs', type=int, default=5, metavar='N',
                   help='epochs to warmup LR, if scheduler supports')
group.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                   help='LR decay rate (default: 0.1)')

# Augmentation & regularization parameters
group = parser.add_argument_group('Augmentation and regularization parameters')
group.add_argument('--no-aug', action='store_true', default=False,
                   help='Disable all training augmentation, override other train aug args')
group.add_argument('--train-crop-mode', type=str, default=None,
                   help='Crop-mode in train')
group.add_argument('--hflip', type=float, default=0.5,
                   help='Horizontal flip training aug probability')
group.add_argument('--scale', type=float, nargs='+', default=[0.08, 1.0], metavar='PCT',
                   help='Random resize scale (default: 0.08 1.0)')

# Miscellaneous parameters
group = parser.add_argument_group('Miscellaneous parameters')
group.add_argument('--seed', type=int, default=42, metavar='S',
                   help='random seed (default: 42)')
group.add_argument('--worker-seeding', type=str, default='all',
                   help='worker seed mode (default: all)')
group.add_argument('--log-interval', type=int, default=50, metavar='N',
                   help='how many batches to wait before logging training status')
group.add_argument('--checkpoint-hist', type=int, default=10, metavar='N',
                   help='number of checkpoints to keep (default: 10)')
group.add_argument('-j', '--workers', type=int, default=8, metavar='N',
                   help='how many training processes to use (default: 4)')
group.add_argument('--no-prefetcher', action='store_true', default=False,
                   help='disable fast prefetcher')
group.add_argument('--output', default='', type=str, metavar='PATH',
                   help='path to output folder (default: none, current dir)')
group.add_argument('--experiment', default='', type=str, metavar='NAME',
                   help='name of train experiment, name of sub-folder for output')
group.add_argument('--use-multi-epochs-loader', action='store_true', default=False,
                   help='use the multi-epochs-loader to save time at the beginning of every epoch')

def _parse_args():
    # Parse config file if provided.
    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)
    args = parser.parse_args(remaining)
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    return args, args_text

def pad_to_size(a, size):
    # same as your original
    current_size = (a.shape[-2], a.shape[-1])
    total_pad_h = size[0] - current_size[0]
    pad_top = total_pad_h // 2
    pad_bottom = total_pad_h - pad_top

    total_pad_w = size[1] - current_size[1]
    pad_left = total_pad_w // 2
    pad_right = total_pad_w - pad_left

    a = nn.functional.pad(a, (pad_left, pad_right, pad_top, pad_bottom))
    return a

# Single-model random_rescale (like your old function). 
def random_rescale(x, scale_bands=4):
    scale_factor_list = np.arange(-scale_bands // 2 + 1, scale_bands // 2 + 2)
    scale_factor_list = [2 ** (i / 4) for i in scale_factor_list]
    scale_factor = random.choice(scale_factor_list)

    img_hw = x.shape[-1]  # Assuming square input
    new_hw = int(img_hw * scale_factor)

    x_rescaled = F.interpolate(x, size=(new_hw, new_hw), mode='bilinear', align_corners=False)
    if new_hw <= img_hw:
        x_rescaled = pad_to_size(x_rescaled, (img_hw, img_hw))
    else:
        center_crop = torchvision.transforms.CenterCrop(img_hw)
        x_rescaled = center_crop(x_rescaled)
    return x_rescaled

def scale_bands_range(bands,new_max,new_min=0,  old_min=0, old_max=10):
    """
    Maps integer 'bands' in [old_min..old_max] to a new range [new_min..new_max].
    If you want discrete buckets, you can add rounding or integer division.
    """
    bands = torch.tensor(bands)
    old_range = old_max - old_min
    new_range = new_max - new_min
    
    # convert to float for safe division, then scale
    scaled = (bands - old_min) / old_range * new_range + new_min
    
    # For integer buckets, round or floor/ceil as desired:
    scaled = torch.floor(scaled)
    
    # convert back to int if needed
    return scaled.long()

def save_image(image, filename):
    # Convert the image to numpy array and transpose to HWC format
    image_np = image.cpu().numpy().transpose(1, 2, 0)
    
    # Normalize the image to [0, 255] range
    image_np = (image_np - image_np.min()) / (image_np.max() - image_np.min())
    image_np = (image_np * 255).astype(np.uint8)
    
    # Save the image
    plt.imsave(filename, image_np)
    print(f"Saved image to {filename}")

def get_scale_band(paths):
    scale_band = []
    # Flatten the nested list structure
    
    for path in paths[0][0]:
        item = path.split('/')[-1]
        if item in lookup_file_scale:
            scale = lookup_file_scale[item]
        else:
            scale = 2
        scale_band.append(scale)
    return scale_band

def train_one_epoch(
        epoch,
        model,
        loader,
        optimizer,
        loss_fn,
        args,
        device=torch.device('cuda'),
        lr_scheduler=None,
        output_dir=None,
        loss_scaler=None,
        model_ema=None,
        mixup_fn=None,
        num_updates_total=None,
):
    
    alpha = args.alpha
    running_loss = 0.
    last_loss = 0.

    
    second_order = False
    update_time_m = utils.AverageMeter()
    data_time_m = utils.AverageMeter()
    losses_m = utils.AverageMeter()
    scale_losses_m = utils.AverageMeter()
    model.train()
    

    accum_steps = 1 #args.grad_accum_steps
    updates_per_epoch = (len(loader) + accum_steps - 1) // accum_steps
    num_updates = epoch * updates_per_epoch
    last_batch_idx = len(loader) - 1

    data_start_time = update_start_time = time.time()
    optimizer.zero_grad()
    update_sample_count = 0
    for batch_idx, (input, target, paths) in enumerate(loader):

        scale_band = get_scale_band(paths)
        # save one image from the batch for debugging
        if batch_idx == 0:
            save_image(input[0], f"input_{batch_idx}.png")
        
        num_bands = args.scale_bands
        
        input = input.to(device)
        target = target.to(device)
        #scale_band = scale_band.to(device)
        #scale_band = scale_band  # So larger scale_band is smaller loss
       
        scale_band = scale_bands_range(scale_band,new_min=0, new_max=num_bands)
        scale_band = num_bands - scale_band
        #center = center.to(device)
        
        last_batch = batch_idx == last_batch_idx
        need_update = True #last_batch or (batch_idx + 1) % accum_steps == 0
        update_idx = batch_idx // accum_steps   
        
        def _forward():
            scale_loss = 0
            
            try:
                if model.module.contrastive_loss:
                    
                    output, scale_loss= model(input,scale_band)
                    scale_loss = args.cl_lambda*scale_loss
                    loss = loss_fn(output, target)  + (scale_loss)
                else:
                    output = model(input)
                   
                    loss = loss_fn(output, target) 
            except Exception as e:
                if model.contrastive_loss:
                    
                    output, scale_loss= model(input,scale_band)
                    
                    scale_loss = args.cl_lambda*scale_loss
                    loss = loss_fn(output, target)  + (scale_loss)
                else:
                    
                    output  = model(input)
                   
                    loss = loss_fn(output, target) 
                    
            #print one sample from scale_logits and its ground truth
            #apply softmax to scale_logits
            #scale_logits = F.softmax(scale_logits, dim=1)
            #print the scale that was selected the most in the batch
            #selected_scale = torch.argmax(scale_logits, dim=1)
            #count_selected_scale = torch.bincount(selected_scale)
            #print(count_selected_scale)
            #count_gt_scale = torch.bincount(scale_band)
            #print(count_gt_scale)
            return loss, scale_loss

        def _backward(_loss):
            if loss_scaler is not None:
                loss_scaler(
                    _loss,
                    optimizer,  
                    clip_grad=args.clip_grad,
                    clip_mode=args.clip_mode,
                    parameters=model_parameters(model, exclude_head='agc' in args.clip_mode),
                    create_graph=second_order,
                    need_update=need_update,
                )
            else:
                _loss.backward(create_graph=second_order)
                if need_update:
                    if args.clip_grad is not None:
                        utils.dispatch_clip_grad(
                            model_parameters(model, exclude_head='agc' in args.clip_mode),
                            value=args.clip_grad,
                            mode=args.clip_mode,
                        )
                    optimizer.step()

        loss,  scale_loss = _forward()
        _backward(loss)

        running_loss += loss.item()
        if batch_idx % 50 == 49:
            last_loss = running_loss / 50
            running_loss = 0.

        if not args.distributed:
            losses_m.update(loss.item() * accum_steps, input.size(0))
            scale_losses_m.update(scale_loss * accum_steps, input.size(0))
        update_sample_count += input.size(0)

        if not need_update:
            data_start_time = time.time()
            continue

        optimizer.zero_grad()

        num_updates += 1
        if model_ema is not None:
            model_ema.update(model, step=num_updates)
            
        if args.synchronize_step and device.type == 'cuda':
            torch.cuda.synchronize()
        time_now = time.time()
        update_time_m.update(time.time() - update_start_time)
        update_start_time = time_now

        if update_idx % args.log_interval == 0:
            #if args.add_wrapped_schedulefree:
            #    lr = optimizer.defaults['lr']  # Get learning rate from ScheduleFree
            #else:
            lrl = [param_group['lr'] for param_group in optimizer.param_groups]
            lr = sum(lrl) / len(lrl)

            if args.distributed:
                reduced_loss = utils.reduce_tensor(loss.data, args.world_size)
                losses_m.update(reduced_loss.item() * accum_steps, input.size(0))   
                scale_losses_m.update(scale_loss.item() * accum_steps, input.size(0))   
                update_sample_count *= args.world_size
            if utils.is_primary(args):
                _logger.info(
                    f'Train: {epoch} [{update_idx:>4d}/{updates_per_epoch} '
                    f'({100. * (update_idx + 1) / updates_per_epoch:>3.0f}%)]  '
                    f'Loss: {losses_m.val:#.3g} ({losses_m.avg:#.3g})  '
                    f'Scale Loss: {scale_losses_m.val:#.3g} ({scale_losses_m.avg:#.3g})  '
                    f'Time: {update_time_m.val:.3f}s  '
                    #f'({update_time_m.avg:.3f}s)  ',
                    f'LR: {lr:.3e}  '
                    f'Data: {data_time_m.val:.3f} ({data_time_m.avg:.3f})'
                )

        if lr_scheduler is not None:
            lr_scheduler.step_update(num_updates=num_updates, metric=losses_m.avg)

        update_sample_count = 0
        data_start_time = time.time()    
    return OrderedDict([
        ('total_loss', losses_m.avg),
        ('scale_loss', scale_losses_m.avg),
    ])

def validate(
    model,
    loader,
    loss_fn,
    args,
    device=torch.device('cuda'),
    log_suffix=''
):
    model.eval()
    
    losses_m = utils.AverageMeter()
    top1_m = utils.AverageMeter()
    top5_m = utils.AverageMeter()
    
    batch_time_m = utils.AverageMeter()
    end = time.time()
    last_idx = len(loader) - 1
    
    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(loader):
            input = input.to(device)
            target = target.to(device)
            if args.channels_last:
                input = input.contiguous(memory_format=torch.channels_last)
            
            # (class_logits, scale_logits) = model(input)
            result = model(input)
            if len(result) == 3:
                output, scale_logits,scale_loss = result
            elif len(result) == 2:
                output, scale_loss = result
            else:
                output = result
                scale_loss = 0
            loss = loss_fn(output, target)
            scale_loss = args.cl_lambda*scale_loss
            loss = loss + scale_loss
            acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
            losses_m.update(loss.item(), input.size(0))
            top1_m.update(acc1.item(), input.size(0))
            top5_m.update(acc5.item(), input.size(0))
            
            batch_time_m.update(time.time() - end)
            end = time.time()
            
            if utils.is_primary(args) and (batch_idx % args.log_interval == 0 or batch_idx == last_idx):
                _logger.info(
                    f'Val{log_suffix}: [{batch_idx}/{last_idx}]  '
                    f'Time: {batch_time_m.val:.3f}s ({batch_time_m.avg:.3f}s)  '
                    f'Loss: {losses_m.val:.4f} ({losses_m.avg:.4f})  '
                    f'Acc@1: {top1_m.val:.3f} ({top1_m.avg:.3f})  '
                    f'Acc@5: {top5_m.val:.3f} ({top5_m.avg:.3f})'
                )
    
    metrics = OrderedDict([
        ('loss', losses_m.avg),
        ('top1', top1_m.avg),
        ('top5', top5_m.avg),
    ])
    return metrics


def main():
    #print(f"Starting main")
    utils.setup_default_logging()
    args, args_text = _parse_args()

    if args.device_modules:
        for module in args.device_modules:
            importlib.import_module(module)

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    args.prefetcher = not args.no_prefetcher
    device = utils.init_distributed_device(args)
    if args.distributed:
        _logger.info(
            'Training in distributed mode with multiple processes, 1 device per process. '
            f'Process {args.rank}, total {args.world_size}, device {args.device}.')
    else:
        _logger.info(f'Training with a single process on 1 device ({args.device}).')
    assert args.rank >= 0

    utils.random_seed(args.seed, args.rank)
    if args.fuser:
        utils.set_jit_fuser(args.fuser)
    if args.fast_norm:
        set_fast_norm()

    in_chans = args.in_chans if args.in_chans is not None else 3
    if args.input_size is not None:
        in_chans = args.input_size[0]

    factory_kwargs = {}
    if args.pretrained_path:
        factory_kwargs['pretrained_cfg_overlay'] = dict(
            file=args.pretrained_path,
            num_classes=-1,
        )

    # Create single model
    model = create_model(
        args.model,
        pretrained=args.pretrained,
        in_chans=in_chans,
        num_classes=args.num_classes,
        checkpoint_path=args.initial_checkpoint,
        **factory_kwargs,
        **args.model_kwargs,
    )
    model = model.to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    _logger.info(f"Model created, param count:{num_params}")
    if args.grad_checkpointing and hasattr(model, 'set_grad_checkpointing'):
        model.set_grad_checkpointing(enable=True)
   
    data_config = resolve_data_config(vars(args), model=model, verbose=utils.is_primary(args))
    if args.data and not args.data_dir:
        args.data_dir = args.data
    input_img_mode = args.input_img_mode if args.input_img_mode is not None else ('RGB' if data_config['input_size'][0] == 3 else 'L')

    dataset_train = create_dataset(
        args.dataset,
        return_paths=True, 
        root=args.data_dir,
        split=args.train_split,
        is_training=True,
        class_map=args.class_map,
        download=args.dataset_download,
        batch_size=args.batch_size,
        seed=args.seed,
        repeats=args.epoch_repeats,
        input_img_mode=input_img_mode,
        input_key=args.input_key,
        target_key=args.target_key,
        num_samples=args.train_num_samples,
    )
    train_interpolation = "bilinear" #args.train_interpolation
    print(args.scale)
    if args.no_aug or not train_interpolation:
        train_interpolation = data_config['interpolation']
    collate_fn = None
    mixup_fn = None
    num_aug_splits = 0
    loader_train = create_loader(
        dataset_train,
        input_size=data_config['input_size'],
        batch_size=args.batch_size,
        is_training=True,
        no_aug=args.no_aug,
        re_prob=0, #args.reprob,
        re_mode='pixel', # shouldn't matter args.remode,
        re_count=0, #args.recount,
        re_split=False, #args.resplit,
        train_crop_mode=args.train_crop_mode,
        scale=args.scale, # [1,1]
        ratio=[1,1],#args.ratio,
        hflip=args.hflip,
        vflip=0, #args.vflip,
        color_jitter=None,#args.color_jitter,
        color_jitter_prob=None, #args.color_jitter_prob,
        grayscale_prob=None, #args.grayscale_prob,
        gaussian_blur_prob=None, #args.gaussian_blur_prob,
        auto_augment=None, #args.aa,
        num_aug_repeats=0, #args.aug_repeats,
        num_aug_splits=num_aug_splits,
        interpolation=train_interpolation,
        mean=data_config['mean'],
        std=data_config['std'],
        num_workers=args.workers,
        distributed=args.distributed,
        collate_fn=collate_fn,
        pin_memory=False, #args.pin_mem,
        device=device,
        use_prefetcher=args.prefetcher,
        use_multi_epochs_loader=args.use_multi_epochs_loader,
        worker_seeding=args.worker_seeding,
    )

    loader_eval = None
    if args.val_split:
        dataset_eval = create_dataset(
            args.dataset,
            root=args.data_dir,
            split=args.val_split,
            is_training=False,
            class_map=args.class_map,
            download=args.dataset_download,
            batch_size=args.batch_size,
            input_img_mode=input_img_mode,
            input_key=args.input_key,
            target_key=args.target_key,
            num_samples=args.val_num_samples,
        )
        eval_workers = args.workers
        if args.distributed and ('tfds' in args.dataset or 'wds' in args.dataset):
            eval_workers = min(2, args.workers)
        loader_eval = create_loader(
            dataset_eval,
            input_size=data_config['input_size'],
            batch_size=args.validation_batch_size or args.batch_size,
            is_training=False,
            interpolation=data_config['interpolation'],
            mean=data_config['mean'],
            std=data_config['std'],
            num_workers=eval_workers,
            distributed=args.distributed,
            crop_pct=data_config['crop_pct'],
            pin_memory=False,
            device=device,
            use_prefetcher=args.prefetcher,
        )

    # Setup loss functions
    train_loss_fn = nn.CrossEntropyLoss().to(device=device)
    validate_loss_fn = nn.CrossEntropyLoss().to(device=device)

    # Create single optimizer
    optimizer = create_optimizer_v2(
        model,
        **optimizer_kwargs(cfg=args),
        **args.opt_kwargs,
    )

    # Create LR scheduler
    updates_per_epoch = (len(loader_train) + args.grad_accum_steps - 1) // args.grad_accum_steps
    lr_scheduler, num_epochs = create_scheduler_v2(
        optimizer,
        **scheduler_kwargs(args, decreasing_metric=True),
        updates_per_epoch=updates_per_epoch,
    )

    start_epoch = 0

    # Setup checkpoint saver
    best_metric = None
    best_epoch = None
    saver = None
    output_dir = None
    if utils.is_primary(args):
        exp_name = args.experiment if args.experiment else '-'.join([
            datetime.now().strftime("%Y%m%d-%H%M%S"),
            safe_model_name(args.model),
            str(data_config['input_size'][-1])
        ])
        output_dir = utils.get_outdir(args.output if args.output else './output/train', exp_name)
        saver = utils.CheckpointSaver(
            model=model,
            optimizer=optimizer,
            args=args,
            amp_scaler=None,
            checkpoint_dir=output_dir,
            recovery_dir=output_dir,
            decreasing=True,
            max_history=args.checkpoint_hist
        )
        with open(os.path.join(output_dir, 'args.yaml'), 'w') as f:
            f.write(args_text)

    results = []
    #try:
    
    if True:
        for epoch in range(start_epoch, num_epochs):
            if hasattr(dataset_train, 'set_epoch'):
                dataset_train.set_epoch(epoch)
            elif args.distributed and hasattr(loader_train.sampler, 'set_epoch'):
                loader_train.sampler.set_epoch(epoch)

            train_metrics = train_one_epoch(
                epoch,
                model,
                loader_train,
                optimizer,
                train_loss_fn,
                args,
                device=device,
                lr_scheduler=lr_scheduler,
            )

            eval_metrics = None
            if loader_eval is not None:
                eval_metrics = validate(
                    model,
                    loader_eval,
                    validate_loss_fn,
                    args,
                    device=device,
                )

            if output_dir is not None:
                lrs = [param_group['lr'] for param_group in optimizer.param_groups]
                utils.update_summary(
                    epoch,
                    train_metrics,
                    eval_metrics,
                    filename=os.path.join(output_dir, 'summary.csv'),
                    lr=sum(lrs) / len(lrs),
                    write_header=best_metric is None,
                    log_wandb=False,
                )

            # If we have validation, pick a metric to track
            latest_metric = eval_metrics['loss'] if eval_metrics is not None else train_metrics['total_loss']
            if saver is not None:
                best_metric, best_epoch = saver.save_checkpoint(epoch, metric=latest_metric)

            if lr_scheduler is not None:
                lr_scheduler.step(epoch + 1, latest_metric)

            results.append({
                'epoch': epoch,
                'train': train_metrics,
                'validation': eval_metrics,
            })

    # except KeyboardInterrupt:
    #     print("KeyboardInterrupt")
        
    #     pass

    results = {'all': results}
    if best_metric is not None and best_epoch is not None:
        results['best'] = results['all'][best_epoch - start_epoch]
        _logger.info('*** Best metric: {0} (epoch {1})'.format(best_metric, best_epoch))
    print(f'--result\n{json.dumps(results, indent=4)}')

if __name__ == '__main__':
    print('before main')
    main()

def create_dataset(
        name,
        root,
        split='validation',
        search_split=True,
        class_map=None,
        load_bytes=False,
        transform=None,
        is_training=False,
        download=False,
        batch_size=None,
        seed=42,
        repeats=0,
        **kwargs
    ):
        """Creates a dataset from name and root.
        """
        name = name.lower()
        if name.startswith('timm/'):
            name = name.split('/', 2)[-1]
            kwargs = {**kwargs, 'root': root, 'split': split, 'download': download}
            dataset = create_dataset(name, **kwargs)
        elif name == 'csv':
            from .parsers import CSVDataset
            dataset = CSVDataset(
                root=root,
                csv_file="/files22_lrsresearch/CLPS_Serre_Lab/projects/prj_concept_surgery/finetuning_models/fp_checked2.csv",
                transform=transform
            )
        else:
            raise RuntimeError(f'Unknown dataset {name}')
        return dataset
