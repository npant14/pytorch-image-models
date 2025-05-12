#!/bin/bash
#SBATCH --time=160:00:00
#SBATCH -p gpu --gres=gpu:8
#SBATCH -n 32
#SBATCH -N 1
#SBATCH --mem=90GB
#SBATCH -o resmax_v4_bypass_cl_0_ip_4_322_18432.out
#SBATCH -e resmax_v4_bypass_cl_0_ip_4_322_18432.err
#SBATCH --account=carney-tserre-condo
#SBATCH -J resmax_v4_bypass_cl_0_ip_4_322_18432


# module load anaconda/2023.09-0-7nso27y
# module load python/3.9.16s-x3wdtvt
# module load cuda

# source  /users/irodri15/data/irodri15/Hmax/hmax_pytorch/venv/bin/activate
cd /users/irodri15/data/irodri15/Hmax/pytorch-image-models/
scale_bands=5
classifier_input_size=18432
model=resmax_v2_sl_bypass
cl_lambda=0.2
bypass=True
sh distributed_train.sh 1 train_skeleton.py \
    --data-dir /gpfs/data/tserre/npant1/ILSVRC/ \
    --dataset torch/imagenet \
    --model $model \
    --model-kwargs ip_scale_bands=$scale_bands classifier_input_size=$classifier_input_size bypass=$bypass\
    --cl-lambda $cl_lambda\
    --opt sgd \
    -b 90 \
    --epochs 100 \
    --lr 1e-2 \
    --weight-decay 5e-4 \
    --sched step \
    --momentum 0.9 \
    --lr-cycle-decay 0.1 \
    --decay-epochs 30 \
    --warmup-epochs 5 \
    --hflip 0.5\
    --scale 1.0 1.0 \
    --start-epoch 0 \
    --workers 8\
    --train-crop-mode rrc\
    --input-size 3 322 322\
    --experiment debug5_resize2_{$model}_bypass_cl_{$cl_lambda}_ip_{$scale_bands}_322_{$classifier_input_size}\
    --output /users/irodri15/data/irodri15/Hmax/pytorch-image-models/output/train/4_25/\
    
 




# # Parameters
# DATASET="torch/imagenet"
# # MODEL="resmax_v2"
# MODEL="chresmax_v2"
# CLASSIFIER_INPUT_SIZE=18432
# CL_LAMBDA=1
# INPUT_SIZE="3 322 322"
# GPUS=8
# IP_BANDS=5
# BATCH_SIZE=32
# EXPERIMENT_NAME="ip_${IP_BANDS}_${MODEL}_gpu_${GPUS}_cl_${CL_LAMBDA}_ip_${INPUT_SIZE// /_}_${CLASSIFIER_INPUT_SIZE}_c1[_6,3,1_]"
# # EXPERIMENT_NAME="test"
# source  /users/irodri15/data/irodri15/Hmax/hmax_pytorch/venv/bin/activate

# sh distributed_train.sh $GPUS train_skeleton.py \
#     --data-dir /gpfs/data/tserre/npant1/ILSVRC/ \
#     --dataset $DATASET \
#     --model $MODEL \
#     --model-kwargs ip_scale_bands=$IP_BANDS classifier_input_size=$CLASSIFIER_INPUT_SIZE bypass=True\
#     --cl-lambda $CL_LAMBDA \
#     --opt sgd \
#     -b $BATCH_SIZE \
#     --epochs 90 \
#     --lr 1e-2 \
#     --weight-decay 5e-4 \
#     --sched step \
#     --momentum 0.9 \
#     --lr-cycle-decay 0.1 \
#     --decay-epochs 30 \
#     --warmup-epochs 0 \
#     --hflip 0.5 \
#     --train-crop-mode rrc \
#     --input-size $INPUT_SIZE \
#     --experiment $EXPERIMENT_NAME \
#     --output /users/irodri15/data/irodri15/Hmax/pytorch-image-models/output/2_25/\