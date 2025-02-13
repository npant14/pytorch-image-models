#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH -p gpu --gres=gpu:8
#SBATCH -n 8
#SBATCH -N 1
#SBATCH --mem=60GB
#SBATCH -o resmax_v1_ip1_bypass.out
#SBATCH -e resmax_v1_ip1_bypass.err
#SBATCH --account=carney-tserre-condo
#SBATCH -J resmax_v1_ip1_bypass
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
# MODEL="vggmax_v1"
MODEL="resmax_v1"
CLASSIFIER_INPUT_SIZE=18432
CL_LAMBDA=0
INPUT_SIZE="3 322 322"
GPUS=8
IP_BANDS=1
BATCH_SIZE=128
EXPERIMENT_NAME="ip_${IP_BANDS}_${MODEL}_gpu_${GPUS}_cl_${CL_LAMBDA}_ip_${INPUT_SIZE// /_}_${CLASSIFIER_INPUT_SIZE}_c1[_6,3,1_]"
# EXPERIMENT_NAME="test"

sh distributed_train.sh $GPUS train_skeleton.py \
    --data-dir /gpfs/data/tserre/npant1/ILSVRC/ \
    --dataset $DATASET \
    --model $MODEL \
    --model-kwargs ip_scale_bands=$IP_BANDS classifier_input_size=$CLASSIFIER_INPUT_SIZE c_scoring="v2" bypass=True \
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
    --train-crop-mode rrc \
    --input-size $INPUT_SIZE \
    --experiment $EXPERIMENT_NAME \
    --output /oscar/data/tserre/xyu110/pytorch-output/train/ \