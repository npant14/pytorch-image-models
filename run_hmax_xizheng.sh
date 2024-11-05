#!/bin/bash
#SBATCH --time=168:00:00
#SBATCH -p gpu --gres=gpu:2
#SBATCH -n 4
#SBATCH -N 1
#SBATCH --mem=80GB
#SBATCH -o hmax_ip_2.out
#SBATCH -e hmax_ip_2.err
#SBATCH --account=carney-tserre-condo
#SBATCH -J hmax_ip_2
#SBATCH --mail-user=xizheng_yu@brown.edu
#SBATCH --mail-type=END,FAIL

module load miniconda3/23.11.0s
source /oscar/runtime/software/external/miniconda3/23.11.0/etc/profile.d/conda.sh
conda activate default

cd /users/xyu110/pytorch-image-models

sh distributed_train.sh 2 train_skeleton.py \
    --data-dir /gpfs/data/tserre/npant1/ILSVRC/ \
    --dataset torch/imagenet \
    --model chmax \
    --model-kwargs ip_scale_bands=2 classifier_input_size=6400 hmax_type='bypass' contrastive_loss=False\
    --opt sgd \
    -b 128 \
    --epochs 90 \
    --lr 1e-2 \
    --weight-decay 5e-4 \
    --sched step \
    --momentum 0.9 \
    --lr-cycle-decay 0.1 \
    --decay-epochs 30 \
    --warmup-epochs 0 \
    --hflip 0.5\
    --train-crop-mode rrc\
    --input-size 3 227 227\
    --scale 1.0 1.0 \
    --experiment hmax_ip_2 \
    --output /gpfs/data/tserre/xyu110/pytorch-output/train \