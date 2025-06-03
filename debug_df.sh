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


MODEL=chresmax_v3_2_dl
echo "Starting experiment"
IP=5 
CL_LAMBDA=0.5
sh distributed_stu_dl.sh 8 train_skeleton_dl.py \
    --data-dir /gpfs/data/tserre/npant1/ILSVRC/ \
    --dataset torch/imagenet \
    --model $MODEL \
    --model-kwargs ip_scale_bands=$IP classifier_input_size=18432 bypass=True contrastive_loss=True   \
    --scale-bands $IP \
    --opt sgd \
    -b 20 \
    --epochs 90 \
    --lr 0.01 \
    --weight-decay 5e-4 \
    --sched step \
    --momentum 0.01 \
    --lr-cycle-decay 0.1 \
    --cl-lambda $CL_LAMBDA \
    --decay-epochs 30 \
    --warmup-epochs 0 \
    --scale 1.0 1.0 \
    --hflip 0.5\
    --workers 4\
    --train-crop-mode rrc\
    --input-size 3 322 322\
    --experiment 29_st1_debug_dl_${MODEL}_${IP}_cl_1_ip_${IP}_322_18432_cl_lambda_${CL_LAMBDA}_lr_0.01 \
    --output /users/irodri15/data/irodri15/Hmax/pytorch-image-models/output/5_25/\
    
 
