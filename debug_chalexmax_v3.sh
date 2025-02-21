#!/bin/bash
#SBATCH --time=160:00:00
#SBATCH -p gpu --gres=gpu:2
#SBATCH -n 4
#SBATCH -N 1
#SBATCH --mem=80GB
#SBATCH -o alexmax_cl_0_ip_5.out
#SBATCH -e alexmax_cl_0_ip_5.err
#SBATCH --account=carney-tserre-condo
#SBATCH -J alexmax_cl_0_ip_5_2


module load anaconda/2023.09-0-7nso27y
module load python/3.9.16s-x3wdtvt
module load cuda

source  /users/irodri15/data/irodri15/Hmax/hmax_pytorch/venv/bin/activate

scale_bands=7
classifier_input_size=9216
model=chalexmax_v3_3_optimized
cl_lambda=1

sh distributed_train.sh 2 train_skeleton.py \
    --data-dir /gpfs/data/tserre/npant1/ILSVRC/ \
    --dataset torch/imagenet \
    --model $model \
    --model-kwargs ip_scale_bands=$scale_bands classifier_input_size=$classifier_input_size\
    --cl-lambda $cl_lambda\
    --opt sgd \
    -b 128 \
    --epochs 120 \
    --lr 1e-2 \
    --weight-decay 5e-4 \
    --sched step \
    --momentum 0.9 \
    --lr-cycle-decay 0.1 \
    --decay-epochs 30 \
    --warmup-epochs 0 \
    --hflip 0.5\
    --train-crop-mode rrc\
    --input-size 3 322 322\
    --experiment debug5_resize2_{$model}_cl_{$cl_lambda}_ip_{$scale_bands}_322_{$classifier_input_size}\
    --output /users/irodri15/data/irodri15/Hmax/pytorch-image-models/output/2_25/\
    
 