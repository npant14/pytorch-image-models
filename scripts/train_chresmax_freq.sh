#!/bin/bash
#SBATCH --time=120:00:00
#SBATCH -p gpu --gres=gpu:8
#SBATCH -n 2
#SBATCH -N 1
#SBATCH --mem=192GB
#SBATCH -o chresmax.out
#SBATCH -e chresmax.err
#SBATCH --account=carney-tserre-condo
#SBATCH -J augmented_chresmax
#SBATCH --mail-user=vatsala_nema@brown.edu
#SBATCH --mail-type=END,FAIL

# wait 10 seconds
sleep 10

# Parameters
DATASET="/gpfs/data/tserre/npant1/ILSVRC/"
MODEL="chresmax_v3_2"
CLASSIFIER_INPUT_SIZE=18432
CL_LAMBDA=0.1
INPUT_SIZE="3 322 322"
GPUS=1
IP_BANDS=3
BATCH_SIZE=128
SCALE=1.0 
BYPASS=True
BYPASS_STR=1.0
LR=0.05
BAND_LOW=16
BAND_HIGH=32
WORKERS=4
EXPERIMENT_NAME="ip_${IP_BANDS}_${MODEL}_gpu_${GPUS}_cl_${CL_LAMBDA}_ip_${INPUT_SIZE// /_}_${CLASSIFIER_INPUT_SIZE}_c1[_6,3,1_]_${SCALE}_bandaug_${BAND_LOW}_${BAND_HIGH}"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Print PyTorch version
python -c "import torch; print('>> TORCH VERSION:', torch.__version__)"

# Run distributed training
sh distributed_train_bandpass.sh $GPUS train_skeleton_bandpass.py \
    --data-dir /gpfs/data/tserre/npant1/ILSVRC/ \
    --dataset torch/imagenet \
    --model $MODEL \
    --model-kwargs ip_scale_bands=$IP_BANDS classifier_input_size=$CLASSIFIER_INPUT_SIZE bypass=$BYPASS contrastive_loss=True \
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
    --use-bandpass \
    --bandpass-low $BAND_LOW \
    --bandpass-high $BAND_HIGH \
    --scale 1.0 1.0 \
    --train-crop-mode rrc \
    --input-size $INPUT_SIZE \
    --experiment $EXPERIMENT_NAME \
    --output output/5_25/$EXPERIMENT_NAME

