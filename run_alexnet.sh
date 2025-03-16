#!/bin/bash
#SBATCH --time=99:00:00
#SBATCH -p gpu --gres=gpu:2
#SBATCH -n 4
#SBATCH -N 1
#SBATCH --mem=80GB
#SBATCH -o alex5.out
#SBATCH -e alex5.err
#SBATCH --account=carney-tserre-condo
#SBATCH -J alex5


module load anaconda/2023.09-0-7nso27y
module load python/3.9.16s-x3wdtvt
module load cuda

source activate /users/npant1/anaconda3/envs/hmax

sh distributed_train.sh 2 train_skeleton.py \
    --data-dir /gpfs/data/tserre/npant1/ILSVRC/ \
    --dataset torch/imagenet \
    --model alexnet \
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
    --experiment alexnet_paper_match_v12 \
    --output /gpfs/data/tserre/npant1/pytorch-output/train \

# sh distributed_train.sh 8 validate.py \
#     --epochs 1 \
#     --data-dir /gpfs/data/tserre/npant1/ILSVRC/ \
#     --dataset torch/imagenet \
#     --model alexnet \
#     --pretrained \
#     -b 128 \