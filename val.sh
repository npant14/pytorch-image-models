#!/bin/bash
#SBATCH --time=3:00:00
#SBATCH -p gpu --gres=gpu:2
#SBATCH -n 2
#SBATCH -N 1
#SBATCH --mem=60GB
#SBATCH -J validations
#SBATCH -o validation.out
#SBATCH -e validation.err
#SBATCH --account=carney-tserre-condo
#SBATCH --mail-user=xizheng_yu@brown.edu
#SBATCH --mail-type=END,FAIL


module load miniconda3/23.11.0s
source /oscar/runtime/software/external/miniconda3/23.11.0/etc/profile.d/conda.sh
conda activate env_default

cd /users/xyu110/pytorch-image-models

DATASET="torch/imagenet"
MODEL="alexmax_v3.1"
CLASSIFIER_INPUT_SIZE=18432
CL_LAMBDA=0
INPUT_SIZE="3 322 322"
GPUS=2
BATCH_SIZE=128

# add for loop to run multiple scales
for imgscale in 160 192 227 270 322
do
    for IP_BANDS in 3
    do
        sh distributed_val.sh $GPUS validate.py \
            --data-dir /gpfs/data/tserre/data/ImageNet/ILSVRC/Data/CLS-LOC \
            --model $MODEL \
            -b $BATCH_SIZE \
            --model-kwargs ip_scale_bands=$IP_BANDS classifier_input_size=$CLASSIFIER_INPUT_SIZE c_scoring="v2" bypass=True \
            --image-scale 3 $imgscale $imgscale \
            --input-size $INPUT_SIZE \
            --pretrained \
            --checkpoint /cifs/data/tserre_lrs/projects/prj_hmax/models/resize2_alexmax_v3.1_cl_1_ip_{3}_322_12544/last.pth.tar \
            --results-file /oscar/data/tserre/xyu110/pytorch-output/train/batch_size_128/${imgscale}_ip_${IP_BANDS}_${MODEL}_gpu_8_cl_0_ip_3_322_322_${CLASSIFIER_INPUT_SIZE}_c1[_6,3,1_].txt
        wait
    done
done


# --checkpoint /oscar/data/tserre/xyu110/pytorch-output/train/ip_${IP_BANDS}_${MODEL}_gpu_8_cl_0_ip_3_322_322_${CLASSIFIER_INPUT_SIZE}_c1[_6,3,1_]/last.pth.tar \