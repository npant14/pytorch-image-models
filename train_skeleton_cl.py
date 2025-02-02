#!/usr/bin/env python3
""" ImageNet Teacher-Student Training Script

This script demonstrates training two networks – a teacher and a student – built from the same
architecture but configurable via separate keyword arguments. The teacher network receives the
original sample while the student network receives a rescaled version of the sample. A KL divergence
loss is computed between their outputs.
  
Extra command-line arguments:
  --teacher-kwargs: extra kwargs for the teacher model.
  --student-kwargs: extra kwargs for the student model.
  --student-scale-bands: an integer controlling the scale bands (see code for details).
  --kl-lambda: weight for the KL divergence loss between teacher and student.
  
For the student rescaling, a simple function randomly selects a scale factor from a set computed as:
  
    scale_factor_list = np.arange(-student_scale_bands//2 + 1, student_scale_bands//2 + 2)
    scale_factor_list = [2**(i/4) for i in scale_factor_list]
  
and then uses bilinear interpolation.
  
Hacked together by / Copyright 2020 Ross Wightman, modified by ChatGPT.
"""
import argparse
import importlib
import json
import logging
import os
import sys
import time
import random
from collections import OrderedDict
from contextlib import suppress
from datetime import datetime
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils
import yaml
from torch.nn.parallel import DistributedDataParallel as NativeDDP

from timm import utils
from timm.data import create_dataset, create_loader, resolve_data_config, AugMixDataset
from timm.layers import convert_sync_batchnorm, set_fast_norm
from timm.models import create_model, safe_model_name, resume_checkpoint, model_parameters
from timm.optim import create_optimizer_v2, optimizer_kwargs
from timm.scheduler import create_scheduler_v2, scheduler_kwargs
from timm.utils import ApexScaler, NativeScaler

try:
    from apex import amp
    from apex.parallel import DistributedDataParallel as ApexDDP
    from apex.parallel import convert_syncbn_model
    has_apex = True
except ImportError:
    has_apex = False

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

has_compile = hasattr(torch, 'compile')

_logger = logging.getLogger('train')

# First arg parser: for the config file.
config_parser = argparse.ArgumentParser(description='Training Config', add_help=False)
config_parser.add_argument('-c', '--config', default='', type=str, metavar='FILE',
                           help='YAML config file specifying default arguments')

parser = argparse.ArgumentParser(description='PyTorch Teacher-Student Training')

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
                   help='Load this checkpoint as if they were the pretrained weights (with adaptation).')
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
# New kwargs for teacher and student models.
group.add_argument('--teacher-kwargs', nargs='*', default={}, action=utils.ParseKwargs,
                   help='Extra kwargs for teacher model.')
group.add_argument('--student-kwargs', nargs='*', default={}, action=utils.ParseKwargs,
                   help='Extra kwargs for student model.')
# New arguments for student scaling and KL divergence.
group.add_argument('--student-scale-bands', type=int, default=4,
                   help='Integer that controls the range of scaling factors for the student input.')
group.add_argument('--kl-lambda', type=float, default=1.0,
                   help='Weight for the KL divergence loss between student and teacher outputs.')

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
group.add_argument('-j', '--workers', type=int, default=4, metavar='N',
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

# --- Helper function for student rescaling ---
def random_rescale(input_tensor, student_scale_bands):
    """
    Given an input batch tensor [B, C, H, W], randomly rescale the images.
    The scale factors are computed as:
      scale_factor_list = np.arange(-student_scale_bands//2 + 1, student_scale_bands//2 + 2)
      scale_factor_list = [2**(i/4) for i in scale_factor_list]
    """
    scale_factors = np.arange(-student_scale_bands//2 + 1, student_scale_bands//2 + 2)
    scale_factors = [2**(i/4) for i in scale_factors]
    factor = random.choice(scale_factors)
    new_h = int(input_tensor.size(2) * factor)
    new_w = int(input_tensor.size(3) * factor)
    return F.interpolate(input_tensor, size=(new_h, new_w), mode='bilinear', align_corners=False)

# --- Training loop for teacher and student ---
def train_one_epoch(
        epoch,
        teacher_model,
        student_model,
        loader,
        teacher_optimizer,
        student_optimizer,
        loss_fn,
        kl_loss_fn,
        args,
        device=torch.device('cuda'),
        lr_scheduler_teacher=None,
        lr_scheduler_student=None,
):
    teacher_model.train()
    student_model.train()
    losses_m = utils.AverageMeter()
    batch_time_m = utils.AverageMeter()

    end = time.time()
    for batch_idx, (input, target) in enumerate(loader):
        # Move input and target to device.
        input = input.to(device)
        target = target.to(device)
        
        # Teacher gets original sample.
        teacher_output = teacher_model(input)
        # Student receives a randomly rescaled version.
        student_input = random_rescale(input, args.student_scale_bands)
        student_output = student_model(student_input)
        
        # Compute losses.
        teacher_loss = loss_fn(teacher_output, target)
        student_loss = loss_fn(student_output, target)
        kl_loss = kl_loss_fn(F.log_softmax(student_output, dim=1),
                             F.softmax(teacher_output, dim=1))
        total_loss = teacher_loss + student_loss + args.kl_lambda * kl_loss

        total_loss.backward()
        teacher_optimizer.step()
        student_optimizer.step()
        teacher_optimizer.zero_grad()
        student_optimizer.zero_grad()

        losses_m.update(total_loss.item(), input.size(0))
        batch_time_m.update(time.time() - end)
        end = time.time()

        if utils.is_primary(args) and batch_idx % args.log_interval == 0:
            lrl = [param_group['lr'] for param_group in teacher_optimizer.param_groups]
            lr = sum(lrl) / len(lrl)
            _logger.info(
                f'Train Epoch: {epoch} [{batch_idx}/{len(loader)}]  '
                f'Loss: {losses_m.val:.4f} ({losses_m.avg:.4f})  '
                f'LR: {lr:.3e}  '
                f'Time: {batch_time_m.val:.3f}s'
            )
    return OrderedDict([('loss', losses_m.avg)])

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

            output = model(input)
            if isinstance(output, (tuple, list)):
                output = output[0]
            loss = loss_fn(output, target)
            acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
            losses_m.update(loss.item(), input.size(0))
            top1_m.update(acc1.item(), input.size(0))
            top5_m.update(acc5.item(), input.size(0))
            batch_time_m.update(time.time() - end)
            end = time.time()
            if utils.is_primary(args) and (batch_idx % args.log_interval == 0 or batch_idx == last_idx):
                _logger.info(
                    f'Test{log_suffix}: [{batch_idx}/{last_idx}]  '
                    f'Time: {batch_time_m.val:.3f}s ({batch_time_m.avg:.3f}s)  '
                    f'Loss: {losses_m.val:.4f} ({losses_m.avg:.4f})  '
                    f'Acc@1: {top1_m.val:.3f} ({top1_m.avg:.3f})  '
                    f'Acc@5: {top5_m.val:.3f} ({top5_m.avg:.3f})'
                )

    metrics = OrderedDict([('loss', losses_m.avg), ('top1', top1_m.avg), ('top5', top5_m.avg)])
    return metrics

def main():
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

    # Create teacher and student models.
    teacher_model = create_model(
        args.model,
        pretrained=args.pretrained,
        in_chans=in_chans,
        num_classes=args.num_classes,
        checkpoint_path=args.initial_checkpoint,
        **factory_kwargs,
        **args.teacher_kwargs,
    )
    student_model = create_model(
        args.model,
        pretrained=args.pretrained,
        in_chans=in_chans,
        num_classes=args.num_classes,
        checkpoint_path=args.initial_checkpoint,
        **factory_kwargs,
        **args.student_kwargs,
    )
    num_params_teacher = sum(p.numel() for p in teacher_model.parameters() if p.requires_grad)
    num_params_student = sum(p.numel() for p in student_model.parameters() if p.requires_grad)
    print("Teacher model parameters:", num_params_teacher)
    print("Student model parameters:", num_params_student)
    print(teacher_model)
    print(student_model)

    if args.grad_checkpointing:
        teacher_model.set_grad_checkpointing(enable=True)
        student_model.set_grad_checkpointing(enable=True)

    data_config = resolve_data_config(vars(args), model=teacher_model, verbose=utils.is_primary(args))

    if args.data and not args.data_dir:
        args.data_dir = args.data
    input_img_mode = args.input_img_mode if args.input_img_mode is not None else ('RGB' if data_config['input_size'][0] == 3 else 'L')

    dataset_train = create_dataset(
        args.dataset,
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
    loader_train = create_loader(
        dataset_train,
        input_size=data_config['input_size'],
        batch_size=args.batch_size,
        is_training=True,
        no_aug=args.no_aug,
        train_crop_mode=args.train_crop_mode,
        hflip=args.hflip,
        interpolation=data_config['interpolation'],
        mean=data_config['mean'],
        std=data_config['std'],
        num_workers=args.workers,
        distributed=args.distributed,
        pin_memory=False,
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

    # Setup loss functions.
    train_loss_fn = nn.CrossEntropyLoss().to(device=device)
    validate_loss_fn = nn.CrossEntropyLoss().to(device=device)
    kl_loss_fn = nn.KLDivLoss(reduction='batchmean').to(device=device)

    # Create separate optimizers for teacher and student.
    teacher_optimizer = create_optimizer_v2(
        teacher_model,
        **optimizer_kwargs(cfg=args),
        **args.opt_kwargs,
    )
    student_optimizer = create_optimizer_v2(
        student_model,
        **optimizer_kwargs(cfg=args),
        **args.opt_kwargs,
    )

    # Create LR schedulers for each optimizer.
    updates_per_epoch = (len(loader_train) + args.grad_accum_steps - 1) // args.grad_accum_steps
    lr_scheduler_teacher, num_epochs = create_scheduler_v2(
        teacher_optimizer,
        **scheduler_kwargs(args, decreasing_metric=True),
        updates_per_epoch=updates_per_epoch,
    )
    lr_scheduler_student, _ = create_scheduler_v2(
        student_optimizer,
        **scheduler_kwargs(args, decreasing_metric=True),
        updates_per_epoch=updates_per_epoch,
    )

    start_epoch = 0

    # Setup checkpoint saver (only on primary process)
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
            model=teacher_model,  # you may save both models if desired
            optimizer=teacher_optimizer,
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
    try:
        for epoch in range(start_epoch, num_epochs):
            if hasattr(dataset_train, 'set_epoch'):
                dataset_train.set_epoch(epoch)
            elif args.distributed and hasattr(loader_train.sampler, 'set_epoch'):
                loader_train.sampler.set_epoch(epoch)

            train_metrics = train_one_epoch(
                epoch,
                teacher_model,
                student_model,
                loader_train,
                teacher_optimizer,
                student_optimizer,
                train_loss_fn,
                kl_loss_fn,
                args,
                device=device,
                lr_scheduler_teacher=lr_scheduler_teacher,
                lr_scheduler_student=lr_scheduler_student,
            )

            eval_metrics = None
            if loader_eval is not None:
                # Validate teacher model only.
                eval_metrics = validate(
                    teacher_model,
                    loader_eval,
                    validate_loss_fn,
                    args,
                    device=device,
                )

            if output_dir is not None:
                lrs = [param_group['lr'] for param_group in teacher_optimizer.param_groups]
                utils.update_summary(
                    epoch,
                    train_metrics,
                    eval_metrics,
                    filename=os.path.join(output_dir, 'summary.csv'),
                    lr=sum(lrs) / len(lrs),
                    write_header=best_metric is None,
                    log_wandb=False,
                )

            latest_metric = eval_metrics['loss'] if eval_metrics is not None else train_metrics['loss']
            if saver is not None:
                best_metric, best_epoch = saver.save_checkpoint(epoch, metric=latest_metric)

            if lr_scheduler_teacher is not None:
                lr_scheduler_teacher.step(epoch + 1, latest_metric)
            if lr_scheduler_student is not None:
                lr_scheduler_student.step(epoch + 1, latest_metric)

            results.append({
                'epoch': epoch,
                'train': train_metrics,
                'validation': eval_metrics,
            })

    except KeyboardInterrupt:
        pass

    results = {'all': results}
    if best_metric is not None:
        results['best'] = results['all'][best_epoch - start_epoch]
        _logger.info('*** Best metric: {0} (epoch {1})'.format(best_metric, best_epoch))
    print(f'--result\n{json.dumps(results, indent=4)}')

if __name__ == '__main__':
    main()
