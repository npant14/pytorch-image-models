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

# Parameters
DATASET="torch/imagenet"
MODEL="alexmax_v2"
BIG_SIZE=322
SMALL_SIZE=227
PYRAMID="True"
CLASSIFIER_INPUT_SIZE=9216
CL_LAMBDA=0
INPUT_SIZE="3 322 322"
EXPERIMENT_NAME="noclamp_2scale_debug_${MODEL}_cl_${CL_LAMBDA}_ip_${INPUT_SIZE// /_}_${CLASSIFIER_INPUT_SIZE}_c1[_6,3,1_]"

sh distributed_train.sh 2 train_skeleton.py \
    --data-dir /gpfs/data/tserre/npant1/ILSVRC/ \
    --dataset $DATASET \
    --model $MODEL \
    --model-kwargs big_size=$BIG_SIZE small_size=$SMALL_SIZE pyramid=$PYRAMID classifier_input_size=$CLASSIFIER_INPUT_SIZE \
    --cl-lambda $CL_LAMBDA \
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
    --hflip 0.5 \
    --train-crop-mode rrc \
    --input-size $INPUT_SIZE \
    --experiment $EXPERIMENT_NAME \
    --output /users/irodri15/data/irodri15/Hmax/pytorch-image-models/output/train/12_24_fix/ \

   