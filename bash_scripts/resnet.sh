#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH --partition=gpu --gres=gpu:8
#SBATCH --account=carney-tserre-condo
#SBATCH -n 8
#SBATCH -N 1
#SBATCH --mem=64G
#SBATCH -o resnet_18_fair_comparasion.out
#SBATCH -e resnet_18_fair_comparasion.err
#SBATCH -J resnet_18_fair_comparasion
#SBATCH --mail-user=xizheng_yu@brown.edu
#SBATCH --mail-type=END,FAIL

which python

cd /users/xyu110/pytorch-image-models

# wait 10 seconds
sleep 10

# Parameters
DATASET="torch/imagenet"
MODEL="resnet18"
INPUT_SIZE="3 322 322"
GPUS=8
IMAGE_SCALE=1.0

BASE_EXPERIMENT_NAME="resnet_18_fair_comparasion"
EXPERIMENT_NAME="${BASE_EXPERIMENT_NAME}"

if [ $IMAGE_SCALE = 0.08 ]; then
    EXPERIMENT_NAME="${BASE_EXPERIMENT_NAME}_scale_${IMAGE_SCALE}"
fi

# Check if directory exists and append suffix if needed
OUTPUT_DIR="/oscar/data/tserre/xyu110/pytorch-output/train/sep"

mkdir -p $OUTPUT_DIR
SUFFIX_COUNT=1

while [ -d "${OUTPUT_DIR}/${EXPERIMENT_NAME}" ]; do
    echo "Directory ${OUTPUT_DIR}/${EXPERIMENT_NAME} already exists, trying with suffix"
    EXPERIMENT_NAME="${BASE_EXPERIMENT_NAME}_${SUFFIX_COUNT}"
    SUFFIX_COUNT=$((SUFFIX_COUNT + 1))
done

echo "Using experiment name: ${EXPERIMENT_NAME}"

# resnet 18
sh ./bash_scripts/distributed_train.sh $GPUS train_skeleton.py \
    --data-dir /gpfs/data/tserre/npant1/ILSVRC/ \
    --dataset $DATASET \
    --model $MODEL \
    --opt sgd \
    -b 32 \
    --epochs 90 \
    --lr 0.1 \
    --weight-decay 1e-4 \
    --sched step \
    --momentum 0.9 \
    --lr-cycle-decay 0.1 \
    --decay-epochs 30 \
    --warmup-epochs 5 \
    --hflip 0.5 \
    --scale $IMAGE_SCALE 1.0 \
    --train-crop-mode rrc \
    --input-size $INPUT_SIZE \
    --experiment $EXPERIMENT_NAME \
    --output $OUTPUT_DIR \
    --workers 8 \
    --add-wrapped-dataloader \
    --model-kwargs ip_scale_bands=1 classifier_input_size=18432 bypass=True
    # --start-epoch 12 \
    # --initial-checkpoint /oscar/data/tserre/xyu110/pytorch-output/train/4/ip_3_resmax_v3_gpu_8_cl_0_ip_3_322_322_512_c1[_6,3,1_]_2/checkpoint-12.pth.tar