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
CL_LAMBDA=0.2
model= 'contrastive_chresmax'
ip_scale_bands=3
ip_scale_bands_student=5

sh distributed_stu_teacher.sh 2 train_skeleton_ts.py \
    --data-dir /gpfs/data/tserre/npant1/ILSVRC/ \
    --dataset torch/imagenet \
    --model $model \
    --model-kwargs ip_scale_bands=$ip_scale_bands ip_scale_bands_student=$ip_scale_bands_student classifier_input_size=18432 bypass=True \
    --opt sgd \
    -b 90 \
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
    --workers 4 \
    --train-crop-mode rrc\
    --input-size 3 322 322\
    --experiment profile_teacher_ip_${ip_scale_bands}_student_ip_${ip_scale_bands_student}_{$model}_bypass_scale_1.0_cl_lambda_${CL_LAMBDA} \
    --output /users/irodri15/data/irodri15/Hmax/pytorch-image-models/output/5_25/\
    
 