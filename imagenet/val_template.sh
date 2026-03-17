#!/bin/bash
#SBATCH --time=TIME_LIMIT
#SBATCH --partition=PARTITION_VALUE --gres=gpu:GPU_COUNT
#SBATCH -n CPU_COUNT
#SBATCH -N 1
#SBATCH --mem=60GB
#SBATCH -o VAL_JOB_NAME.out
#SBATCH -e VAL_JOB_NAME.err
#SBATCH -J VAL_JOB_NAME
#SBATCH --mail-user=xizheng_yu@brown.edu
#SBATCH --mail-type=END,FAIL

cd /users/xyu110/pytorch-image-models

DATASET="torch/imagenet"
MODEL="MODEL_NAME"
EVAL_MODEL="$MODEL"
CLASSIFIER_INPUT_SIZE=CLS_INPUT_SIZE
CL_LAMBDA=CL_LAMBDA_VALUE
INPUT_SIZE="3 322 322"
GPUS=GPU_COUNT
BATCH_SIZE=BATCH_SIZE_VALUE
IP_BANDS=IP_BANDS_VALUE
BYPASS=BYPASS_VALUE
BYPASS_STR=BYPASS_STR_VALUE
IMAGE_SCALE="IMAGE_SCALE_VALUE"
RESULTS_DIR="RESULTS_DIR_VALUE"
CPUS=CPU_COUNT
PADDING_MODE=PADDING_MODE_VALUE

# MODEL_PTH_USED="model_best"
if [ $MODEL = "contrastive_resmaxv1" ]; then
    MODEL_PTH_USED="model_best"
else
    MODEL_PTH_USED="model_best"
fi


if [ "CKPT_DIR_VALUE" != "" ]; then
    CHECKPOINT_PATH="CKPT_DIR_VALUE/${MODEL_PTH_USED}.pth.tar"
fi

# Use TIMM pretrained weights (no local checkpoint), 224x224 input.
# `alexnet_timm` / `resnet18_timm` keep the original model names for custom checkpoint runs.
if [[ "$MODEL" == "alexnet_timm" ]]; then
    EVAL_MODEL="alexnet"
    CHECKPOINT_PATH=""
    INPUT_SIZE="3 224 224"
elif [[ "$MODEL" == "resnet18_timm" ]]; then
    EVAL_MODEL="resnet18.tv_in1k"
    CHECKPOINT_PATH=""
    INPUT_SIZE="3 224 224"
elif [[ "$MODEL" == "vit_base_patch16_224" ]]; then
    EVAL_MODEL="vit_base_patch16_224.orig_in21k_ft_in1k"
    CHECKPOINT_PATH=""
    INPUT_SIZE="3 224 224"
elif [[ "$MODEL" == vit_* ]]; then
    CHECKPOINT_PATH=""
    INPUT_SIZE="3 224 224"
fi

mkdir -p $RESULTS_DIR
# results_file="${RESULTS_DIR}/baseline_waug_${MODEL_PTH_USED}_${PADDING_MODE}.csv"
results_file="validation_imagenet.csv"

# Build optional checkpoint argument
if [ -n "$CHECKPOINT_PATH" ]; then
    CHECKPOINT_ARG="--checkpoint $CHECKPOINT_PATH"
else
    CHECKPOINT_ARG=""
fi

# Only pass HMAX-specific model-kwargs for custom models.
# Skip for TIMM-pretrained variants.
if [[ "$MODEL" == vit_* || "$MODEL" == "alexnet_timm" || "$MODEL" == "resnet18_timm" ]]; then
    MODEL_KWARGS_ARG=""
else
    MODEL_KWARGS_ARG="--model-kwargs ip_scale_bands=$IP_BANDS classifier_input_size=$CLASSIFIER_INPUT_SIZE c_scoring=v2 bypass=$BYPASS cl=$CL_LAMBDA padding_mode=$PADDING_MODE"
fi

# Run validation for specified parameters
sh imagenet/distributed_val.sh $GPUS validate.py \
    --data-dir /gpfs/data/tserre/data/ImageNet/ILSVRC/Data/CLS-LOC \
    --model $EVAL_MODEL \
    -b $BATCH_SIZE \
    $MODEL_KWARGS_ARG \
    --image-scale 3 $IMAGE_SCALE $IMAGE_SCALE \
    --input-size $INPUT_SIZE \
    --pretrained \
    $CHECKPOINT_ARG \
    --results-file $results_file \
    --workers $CPUS

# MNIST
# results_file="${RESULTS_DIR}/mnist.csv"
# sh distributed_val.sh $GPUS validate.py \
#     --data-dir /oscar/data/tserre/xyu110/mnist \
#     --num-classes 10 \
#     --model $MODEL \
#     -b $BATCH_SIZE \
#     --model-kwargs ip_scale_bands=$IP_BANDS classifier_input_size=$CLASSIFIER_INPUT_SIZE c_scoring="v2" bypass=$BYPASS cl=$CL_LAMBDA padding_mode=$PADDING_MODE\
#     --image-scale 3 $IMAGE_SCALE $IMAGE_SCALE \
#     --input-size 3 224 224 \
#     --pretrained \
#     --checkpoint $CHECKPOINT_PATH \
#     --results-file $results_file \
#     --workers $CPUS
