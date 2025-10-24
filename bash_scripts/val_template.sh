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

module load miniconda3/23.11.0s
source /oscar/runtime/software/external/miniconda3/23.11.0/etc/profile.d/conda.sh
# conda activate env_default
conda activate brain_env

cd /users/xyu110/pytorch-image-models

DATASET="torch/imagenet"
MODEL="MODEL_NAME"
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

mkdir -p $RESULTS_DIR
# results_file="${RESULTS_DIR}/baseline_waug_${MODEL_PTH_USED}_${PADDING_MODE}.csv"
results_file="fair_comparasion.csv"

# Run validation for specified parameters
sh bash_scripts/distributed_val.sh $GPUS validate.py \
    --data-dir /gpfs/data/tserre/data/ImageNet/ILSVRC/Data/CLS-LOC \
    --model $MODEL \
    -b $BATCH_SIZE \
    --model-kwargs ip_scale_bands=$IP_BANDS classifier_input_size=$CLASSIFIER_INPUT_SIZE c_scoring="v2" bypass=$BYPASS cl=$CL_LAMBDA padding_mode=$PADDING_MODE\
    --image-scale 3 $IMAGE_SCALE $IMAGE_SCALE \
    --input-size $INPUT_SIZE \
    --pretrained \
    --checkpoint $CHECKPOINT_PATH \
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
