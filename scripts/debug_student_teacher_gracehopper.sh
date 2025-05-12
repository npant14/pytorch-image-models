#!/bin/bash
#SBATCH --time=160:00:00
#SBATCH -p gpu --gres=gpu:2
#SBATCH -n 4
#SBATCH -N 1
#SBATCH --mem=80GB
#SBATCH -o alexmax_cl_0_ip_5.out

source ~/miniforge3/bin/activate
conda activate venv3.11 
CL_LAMBDA=0.1
model=ft_resmax_v4

sh distributed_stu_teacher.sh 1 train_skeleton_ts.py \
    --data-dir /gpfs/data/tserre/npant1/ILSVRC/ \
    --dataset torch/imagenet \
    --model $model \
    --model-kwargs ip_scale_bands=3  ip_scale_bands_student=5 classifier_input_size=18432 bypass=True \
    --opt sgd \
    -b 128 \
    --epochs 30 \
    --cl-lambda $CL_LAMBDA\
    --lr 1e-4 \
    --weight-decay 5e-4 \
    --sched step \
    --momentum 0.9 \
    --lr-cycle-decay 0.1 \
    --decay-epochs 10 \
    --warmup-epochs 0 \
    --scale 0.9 1.1  \
    --hflip 0.5\
    --workers 8 \
    --train-crop-mode rrc\
    --input-size 3 322 322\
    --experiment profile_teacher_ip_3_student_ip_3_resmax_v3_bypass_scale_1.0_cl_lambda_$CL_LAMBDA \
    --output /users/irodri15/data/irodri15/Hmax/pytorch-image-models/output/5_25/\
    
 