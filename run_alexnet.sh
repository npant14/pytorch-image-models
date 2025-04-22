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




V2_iterations=2
V4_iterations=4
IT_iterations=2
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
    --start-epoch 0 \
    --hflip 0.5\
    --train-crop-mode rrc\
    --input-size 3 322 322\
    --scale 1.0 1.0 \
    #--initial-checkpoint /users/irodri15/data/irodri15/Hmax/pytorch-image-models/output/train/4_25/cornet_s_debug_322_0_0_0_0/last.pth.tar \
    --experiment cornet_s_debug_${IM_SIZE}_${V2_iterations}_${V4_iterations}_${IT_iterations} \
    --output /users/irodri15/data/irodri15/Hmax/pytorch-image-models/output/train/4_25/ \