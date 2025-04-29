#!/bin/bash
#SBATCH --time=TIME_LIMIT
#SBATCH --partition=PARTITION_VALUE --gres=gpu:GPU_COUNT
#SBATCH -n CPU_COUNT
#SBATCH -N 1
#SBATCH --mem=MEM_VALUE
#SBATCH -o JOB_NAME.out
#SBATCH -e JOB_NAME.err
#SBATCH -J JOB_NAME
#SBATCH --mail-user=xizheng_yu@brown.edu
#SBATCH --mail-type=END,FAIL

# Check if modules are loaded
if ml 2>&1 | grep -q "No modules loaded"; then
    module load miniconda3/23.11.0s
    source /oscar/runtime/software/external/miniconda3/23.11.0/etc/profile.d/conda.sh
    conda activate env_default
else
    echo "Modules already loaded, skipping conda activation"
fi

which python

cd /users/xyu110/pytorch-image-models

# wait 10 seconds
sleep 10

# Parameters
DATASET="torch/imagenet"
MODEL="MODEL_NAME"
CLASSIFIER_INPUT_SIZE=CLS_INPUT_SIZE
CL_LAMBDA=CL_LAMBDA_VALUE
INPUT_SIZE="3 INPUT_SIZE_VALUE INPUT_SIZE_VALUE"
GPUS=GPU_COUNT
LR=LR_VALUE
IP_BANDS=IP_BANDS_VALUE
BATCH_SIZE=BATCH_SIZE_VALUE
BYPASS=BYPASS_VALUE
BYPASS_STR=BYPASS_STR_VALUE
WORKERS=WORKERS_VALUE
IMAGE_SCALE=IMAGE_SCALE_VALUE

BASE_EXPERIMENT_NAME="ip_${IP_BANDS}_${MODEL}_gpu_${GPUS}_cl_${CL_LAMBDA}_ip_${INPUT_SIZE// /_}_${CLASSIFIER_INPUT_SIZE}_c1[_6,3,1_]${BYPASS_STR}"
EXPERIMENT_NAME="${BASE_EXPERIMENT_NAME}"

if [ $IMAGE_SCALE = 0.08 ]; then
    EXPERIMENT_NAME="${BASE_EXPERIMENT_NAME}_scale_${IMAGE_SCALE}"
fi

# Check if directory exists and append suffix if needed
OUTPUT_DIR="/oscar/data/tserre/xyu110/pytorch-output/train/4"
mkdir -p OUTPUT_DIR
SUFFIX_COUNT=1

while [ -d "${OUTPUT_DIR}/${EXPERIMENT_NAME}" ]; do
    echo "Directory ${OUTPUT_DIR}/${EXPERIMENT_NAME} already exists, trying with suffix"
    EXPERIMENT_NAME="${BASE_EXPERIMENT_NAME}_${SUFFIX_COUNT}"
    SUFFIX_COUNT=$((SUFFIX_COUNT + 1))
done

echo "Using experiment name: ${EXPERIMENT_NAME}"

sh distributed_train.sh $GPUS train_skeleton.py \
    --data-dir /gpfs/data/tserre/npant1/ILSVRC/ \
    --dataset $DATASET \
    --model $MODEL \
    --model-kwargs ip_scale_bands=$IP_BANDS classifier_input_size=$CLASSIFIER_INPUT_SIZE bypass=$BYPASS\
    --cl-lambda $CL_LAMBDA \
    --opt sgd \
    -b $BATCH_SIZE \
    --epochs 90 \
    --lr $LR \
    --weight-decay 5e-4 \
    --sched step \
    --momentum 0.9 \
    --lr-cycle-decay 0.1 \
    --decay-epochs 30 \
    --warmup-epochs 0 \
    --hflip 0.5 \
    --scale $IMAGE_SCALE 1.0 \
    --train-crop-mode rrc \
    --input-size $INPUT_SIZE \
    --experiment $EXPERIMENT_NAME \
    --output $OUTPUT_DIR

# resnet 18
# sh distributed_train.sh $GPUS train_skeleton.py \
#     --data-dir /gpfs/data/tserre/npant1/ILSVRC/ \
#     --dataset $DATASET \
#     --model $MODEL \
#     --model-kwargs ip_scale_bands=$IP_BANDS classifier_input_size=$CLASSIFIER_INPUT_SIZE bypass=$BYPASS\
#     --opt sgd \
#     -b $BATCH_SIZE \
#     --epochs 90 \
#     --lr $LR \
#     --weight-decay 1e-4 \
#     --sched step \
#     --momentum 0.9 \
#     --lr-cycle-decay 0.1 \
#     --decay-epochs 30 \
#     --warmup-epochs 5 \
#     --hflip 0.5 \
#     --scale $IMAGE_SCALE 1.0 \
#     --train-crop-mode rrc \
#     --input-size $INPUT_SIZE \
#     --experiment $EXPERIMENT_NAME \
#     --output $OUTPUT_DIR \
#     --workers $WORKERS \
#     # --start-epoch 12 \
#     # --initial-checkpoint /oscar/data/tserre/xyu110/pytorch-output/train/4/ip_3_resmax_v3_gpu_8_cl_0_ip_3_322_322_512_c1[_6,3,1_]_2/checkpoint-12.pth.tar


# alexnet
# sh distributed_train.sh $GPUS train_skeleton.py \
#     --data-dir /gpfs/data/tserre/npant1/ILSVRC/ \
#     --dataset $DATASET \
#     --model $MODEL \
#     --model-kwargs ip_scale_bands=$IP_BANDS \
#     --opt sgd \
#     -b 128 \
#     --epochs 90 \
#     --lr 0.01 \
#     --weight-decay 5e-4 \
#     --sched step \
#     --decay-epochs 30 \
#     --decay-rate 0.1 \
#     --lr-cycle-decay 0.1 \
#     --momentum 0.9 \
#     --warmup-epochs 0 \
#     --hflip 0.5 \
#     --scale $IMAGE_SCALE 1.0 \
#     --train-crop-mode rrc \
#     --input-size $INPUT_SIZE \
#     --experiment $EXPERIMENT_NAME \
#     --output $OUTPUT_DIR \
#     --workers $WORKERS

# vgg
# sh distributed_train.sh $GPUS train_skeleton.py \
#     --data-dir /gpfs/data/tserre/npant1/ILSVRC/ \
#     --dataset $DATASET \
#     --model $MODEL \
#     --model-kwargs ip_scale_bands=$IP_BANDS \
#     --opt sgd \
#     -b 32 \
#     --epochs 75 \
#     --lr 0.01 \
#     --weight-decay 5e-4 \
#     --sched step \
#     --momentum 0.9 \
#     --lr-cycle-decay 0.1 \
#     --decay-epochs 25 \
#     --warmup-epochs 0 \
#     --hflip 0.5 \
#     --scale $IMAGE_SCALE 1.0 \
#     --train-crop-mode rrc \
#     --input-size $INPUT_SIZE \
#     --experiment $EXPERIMENT_NAME \
#     --output $OUTPUT_DIR \
#     --workers $WORKERS