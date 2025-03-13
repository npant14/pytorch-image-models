#!/bin/bash
#SBATCH --time=TIME_LIMIT
#SBATCH -p gpu --gres=gpu:GPU_COUNT
#SBATCH -n GPU_COUNT
#SBATCH -N 1
#SBATCH --mem=60GB
#SBATCH -o JOB_NAME.out
#SBATCH -e JOB_NAME.err
#SBATCH --account=carney-tserre-condo
#SBATCH -J JOB_NAME
#SBATCH --mail-user=xizheng_yu@brown.edu
#SBATCH --mail-type=END,FAIL

module load miniconda3/23.11.0s
source /oscar/runtime/software/external/miniconda3/23.11.0/etc/profile.d/conda.sh
conda activate env_default

cd /users/xyu110/pytorch-image-models

# wait 10 seconds
sleep 10

# Parameters
DATASET="torch/imagenet"
MODEL="MODEL_NAME"
CLASSIFIER_INPUT_SIZE=CLS_INPUT_SIZE
CL_LAMBDA=CL_LAMBDA_VALUE
INPUT_SIZE="3 322 322"
GPUS=GPU_COUNT
IP_BANDS=IP_BANDS_VALUE
BATCH_SIZE=BATCH_SIZE_VALUE
BYPASS=BYPASS_VALUE
BYPASS_STR=BYPASS_STR_VALUE

EXPERIMENT_NAME="ip_${IP_BANDS}_${MODEL}_gpu_${GPUS}_cl_${CL_LAMBDA}_ip_${INPUT_SIZE// /_}_${CLASSIFIER_INPUT_SIZE}_c1[_6,3,1_]${BYPASS_STR}"

sh distributed_train.sh $GPUS train_skeleton.py \
    --data-dir /gpfs/data/tserre/npant1/ILSVRC/ \
    --dataset $DATASET \
    --model $MODEL \
    --model-kwargs ip_scale_bands=$IP_BANDS classifier_input_size=$CLASSIFIER_INPUT_SIZE bypass=$BYPASS \
    --cl-lambda $CL_LAMBDA \
    --opt sgd \
    -b $BATCH_SIZE \
    --epochs 90 \
    --lr 1e-2 \
    --weight-decay 5e-4 \
    --sched step \
    --momentum 0.9 \
    --lr-cycle-decay 0.1 \
    --decay-epochs 30 \
    --warmup-epochs 0 \
    --hflip 0.5 \
    --scale 1.0 1.0 \
    --train-crop-mode rrc \
    --input-size $INPUT_SIZE \
    --experiment $EXPERIMENT_NAME \
    --output /oscar/data/tserre/xyu110/pytorch-output/train/
    