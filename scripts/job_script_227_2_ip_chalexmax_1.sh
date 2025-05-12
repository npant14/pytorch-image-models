#!/bin/bash
#SBATCH --time=160:00:00
#SBATCH -p gpu --gres=gpu:2
#SBATCH -n 4
#SBATCH -N 1
#SBATCH --mem=80GB
#SBATCH -o results/hmax_chalexmax_ip_1_ip_2.out
#SBATCH -e results/hmax_chalexmax_ip_1_ip_2.err
#SBATCH --account=carney-tserre-condo
#SBATCH -J results/hmax_chalexmax_ip_1_ip_2

module load anaconda/2023.09-0-7nso27y
module load python/3.9.16s-x3wdtvt
module load cuda

source  /users/irodri15/data/irodri15/Hmax/hmax_pytorch/venv/bin/activate

sh distributed_train.sh 2 train_skeleton.py     --data-dir /gpfs/data/tserre/npant1/ILSVRC/     --dataset torch/imagenet     --model chalexmax     --model-kwargs ip_scale_bands=2 classifier_input_size=9216 hmax_type='alexmax'     --cl-lambda 1     --opt sgd     -b 128     --epochs 160     --lr 1e-2     --weight-decay 5e-4     --sched step     --momentum 0.9     --lr-cycle-decay 0.1     --decay-epochs 30     --warmup-epochs 0     --hflip 0.5     --train-crop-mode rrc     --input-size 3 227 227     --experiment chalexmax_227_cl_1_ip_2_nearest     --output /users/irodri15/data/irodri15/Hmax/pytorch-image-models/output/train
