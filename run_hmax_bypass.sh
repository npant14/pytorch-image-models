#!/bin/bash
#SBATCH --time=160:00:00
#SBATCH -p gpu --gres=gpu:2
#SBATCH -n 4
#SBATCH -N 1
#SBATCH --mem=80GB
#SBATCH -o hmax_bypass_cl_0_ip_1.out
#SBATCH -e hmax_bypass_cl_0_ip_1.err
#SBATCH --account=carney-tserre-condo
#SBATCH -J hmax_bypass_cl_0_ip_1


module load anaconda/2023.09-0-7nso27y
module load python/3.9.16s-x3wdtvt
module load cuda

source  /users/irodri15/data/irodri15/Hmax/hmax_pytorch/venv/bin/activate

sh distributed_train.sh 2 train_skeleton.py \
    --data-dir /gpfs/data/tserre/npant1/ILSVRC/ \
    --dataset torch/imagenet \
    --model chmax \
    --model-kwargs ip_scale_bands=3 classifier_input_size=16384 hmax_type='bypass'\
    --cl-lambda 0.1\
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
    --input-size 3 454 454\
    --experiment bypass_cl_0.1_ip_3_454 \
    --output /users/irodri15/data/irodri15/Hmax/pytorch-image-models/pytorch-output/train \
