#!/bin/bash
#SBATCH --time=3:00:00
#SBATCH -p gpu --gres=gpu:2
#SBATCH -n 4
#SBATCH -N 1
#SBATCH --mem=60GB
#SBATCH -J image_alex_size_0.08_validation
#SBATCH -o /users/xyu110/pytorch-image-models/alex0.08_validation.out
#SBATCH -e /users/xyu110/pytorch-image-models/alex0.08_validation.err
#SBATCH --account=carney-tserre-condo

#SBATCH --mail-user=xizheng_yu@brown.edu
#SBATCH --mail-type=END,FAIL


module load miniconda3/23.11.0s
source /oscar/runtime/software/external/miniconda3/23.11.0/etc/profile.d/conda.sh
conda activate env_default

cd /users/xyu110/pytorch-image-models

DATASET="torch/imagenet"
MODEL="resmax_bypass"
BIG_SIZE=322
SMALL_SIZE=227
PYRAMID="True"
CLASSIFIER_INPUT_SIZE=18432
CL_LAMBDA=0
INPUT_SIZE="3 322 322"
GPUS=8
CONV="double3x3"

# add for loop to run multiple scales
for imgscale in 160 192 227 270 322
do
    for alexscale in 322
    do
        if [ $imgscale -le $alexscale ]
        then
            sh distributed_val.sh 2 validate.py \
                --data-dir /gpfs/data/tserre/data/ImageNet/ILSVRC/Data/CLS-LOC \
                --model $MODEL \
                -b 128 \
                --model-kwargs big_size=$BIG_SIZE small_size=$SMALL_SIZE pyramid=$PYRAMID classifier_input_size=$CLASSIFIER_INPUT_SIZE conv_type=$CONV\
                --image-scale 3 $imgscale $imgscale \
                --input-size $INPUT_SIZE \
                --pretrained \
                --checkpoint /oscar/data/tserre/xyu110/pytorch-output/train/c_score/debug_resmax_bypass_concat_gpu_8_cl_0_ip_3_322_322_18432_c1[_6,3,1_]/last.pth.tar \
                --results-file /oscar/data/tserre/xyu110/pytorch-output/train/debug_resmax_bypass_concat_gpu_8_cl_0_ip_3_322_322_18432_c1[_6,3,1_].txt
            wait
        fi
    done
done
