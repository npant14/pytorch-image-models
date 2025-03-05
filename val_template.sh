#!/bin/bash
#SBATCH --time=TIME_LIMIT
#SBATCH -p gpu --gres=gpu:GPU_COUNT
#SBATCH -n GPU_COUNT
#SBATCH -N 1
#SBATCH --mem=60GB
#SBATCH -o VAL_JOB_NAME.out
#SBATCH -e VAL_JOB_NAME.err
#SBATCH --account=carney-tserre-condo
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

mkdir -p $RESULTS_DIR
OUTPUT_FILE_NAME=${RESULTS_DIR}/${IMAGE_SCALE}_ip_${IP_BANDS}_${MODEL}_gpu_${GPUS}_cl_${CL_LAMBDA}_ip_3_322_322_${CLASSIFIER_INPUT_SIZE}_c1[_6,3,1_]${BYPASS_STR}.txt

# if resultfile exists, exit
# if [ -f $OUTPUT_FILE_NAME ]; then
#     echo "Result file ${OUTPUT_FILE_NAME} exists, exiting"
#     exit 0
# fi

# Run validation for specified parameters
sh distributed_val.sh $GPUS validate.py \
    --data-dir /gpfs/data/tserre/data/ImageNet/ILSVRC/Data/CLS-LOC \
    --model $MODEL \
    -b $BATCH_SIZE \
    --model-kwargs ip_scale_bands=$IP_BANDS classifier_input_size=$CLASSIFIER_INPUT_SIZE c_scoring="v2" bypass=$BYPASS brainscore=False\
    --image-scale 3 $IMAGE_SCALE $IMAGE_SCALE \
    --input-size $INPUT_SIZE \
    --pretrained \
    --checkpoint /oscar/data/tserre/xyu110/pytorch-output/train/recent_results/ip_${IP_BANDS}_${MODEL}_gpu_8_cl_${CL_LAMBDA}_ip_3_322_322_${CLASSIFIER_INPUT_SIZE}_c1[_6,3,1_]/model_best.pth.tar \
    --results-file $OUTPUT_FILE_NAME