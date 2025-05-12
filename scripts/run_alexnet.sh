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




V2_iterations=1
V4_iterations=1
IT_iterations=1
IM_SIZE=322
sh distributed_train.sh 2 train_skeleton.py \
    --data-dir /gpfs/data/tserre/npant1/ILSVRC/ \
    --dataset torch/imagenet \
    --model cornet_s \
    --opt sgd \
    -b 128 \
    --epochs 90 \
    --model-kwargs V2_iterations=$V2_iterations V4_iterations=$V4_iterations IT_iterations=$IT_iterations\
    --lr 1e-2 \
    --weight-decay 5e-4 \
    --sched step \
    --momentum 0.9 \
    --lr-cycle-decay 0.1 \
    --decay-epochs 30 \
    --warmup-epochs 0 \
    --start-epoch 9 \
    --hflip 0.5\
    --train-crop-mode rrc\
    --input-size 3 322 322\
    --scale 1.0 1.0 \
    --initial-checkpoint /users/irodri15/data/irodri15/Hmax/pytorch-image-models/output/train/4_25/cornet_s_debug2_322_1_1_1/checkpoint-8.pth.tar \
    --experiment cornet_s_debug2_${IM_SIZE}_${V2_iterations}_${V4_iterations}_${IT_iterations} \
    --output /users/irodri15/data/irodri15/Hmax/pytorch-image-models/output/train/4_25/ \