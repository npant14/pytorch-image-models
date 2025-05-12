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





MODEL=chalexmax_bypass_dl_v2
echo "Starting experiment"
IP=5 
ALPHA=0
CL_LAMBDA=0
sh distributed_stu_dl.sh 2 train_skeleton_dl.py \
    --data-dir /gpfs/data/tserre/npant1/ILSVRC/ \
    --dataset torch/imagenet \
    --model $MODEL \
    --model-kwargs ip_scale_bands=$IP classifier_input_size=9216  \
    --scale-bands $IP \
    --opt sgd \
    -b 64 \
    --epochs 90 \
    --lr 1e-2 \
    --weight-decay 5e-4 \
    --sched step \
    --alpha $ALPHA \
    --momentum 0.9 \
    --lr-cycle-decay 0.1 \
    --cl-lambda $CL_LAMBDA \
    --decay-epochs 30 \
    --warmup-epochs 0 \
    --scale 1.0 1.0 \
    --hflip 0.5\
    --train-crop-mode rrc\
    --input-size 3 322 322\
    --experiment 24_debug_dl_${MODEL}_${IP}_cl_1_ip_${IP}_322_9216_alpha_${ALPHA}_cl_lambda_${CL_LAMBDA} \
    --output /users/irodri15/data/irodri15/Hmax/pytorch-image-models/output/3_25/\
    
 
