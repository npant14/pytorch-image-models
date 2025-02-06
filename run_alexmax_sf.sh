#!/bin/bash
#SBATCH --time=72:00:00
#SBATCH -p gpu --gres=gpu:8
#SBATCH -n 8
#SBATCH -N 1
#SBATCH --mem=60GB
#SBATCH -o schedulefree_resmax.out
#SBATCH -e schedulefree_resmax.err
#SBATCH --account=carney-tserre-condo
#SBATCH -J schedulefree_resmax
#SBATCH --mail-user=xizheng_yu@brown.edu
#SBATCH --mail-type=END,FAIL


module load miniconda3/23.11.0s
source /oscar/runtime/software/external/miniconda3/23.11.0/etc/profile.d/conda.sh
conda activate env_default

cd /users/xyu110/pytorch-image-models

# Parameters
DATASET="torch/imagenet"
MODEL="resmax"
BIG_SIZE=322
SMALL_SIZE=227
PYRAMID="True"
CLASSIFIER_INPUT_SIZE=9216
CL_LAMBDA=0
INPUT_SIZE="3 322 322"
GPUS=8
EXPERIMENT_NAME="${MODEL}_schedulefree_gpu_${GPUS}_cl_${CL_LAMBDA}_ip_${INPUT_SIZE// /_}_${CLASSIFIER_INPUT_SIZE}_c1[_6,3,1_]"

sh distributed_train.sh $GPUS train_skeleton.py \
    --data-dir /gpfs/data/tserre/npant1/ILSVRC/ \
    --dataset $DATASET \
    --model $MODEL \
    --model-kwargs big_size=$BIG_SIZE small_size=$SMALL_SIZE pyramid=$PYRAMID classifier_input_size=$CLASSIFIER_INPUT_SIZE \
    --cl-lambda $CL_LAMBDA \
    --opt sgd \
    -b 128 \
    --epochs 90 \
    --lr 1.5 \
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
    --output /oscar/data/tserre/xyu110/pytorch-output/train/ \
    --add-wrapped-schedulefree \