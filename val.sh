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

# add for loop to run multiple scales
for imgscale in 160 192 227 270 321 382 454
do
    for alexscale in 160 192 227 270 321 382 454
    do
        if [ $imgscale -le $alexscale ]
        then
            sh distributed_val.sh 2 validate.py \
                --data-dir /gpfs/data/tserre/data/ImageNet/ILSVRC/Data/CLS-LOC \
                --model alexnet \
                -b 128 \
                --image-scale 3 $imgscale $imgscale \
                --input-size 3 $alexscale $alexscale \
                --pretrained \
                --checkpoint /oscar/data/tserre/xyu110/pytorch-output/train/alexnet_size_zoom_out_${alexscale}/last.pth.tar \
                --results-file /oscar/data/tserre/xyu110/pytorch-output/train/alexnet_zoom_out/scale_${imgscale}_alex_size_${alexscale}.txt
            wait
        fi
    done
done

# sh distributed_val.sh 2 validate.py \
#     --data-dir /gpfs/data/tserre/data/ImageNet/ILSVRC/Data/CLS-LOC \
#     --model alexnet \
#     -b 128 \
#     --input-size 3 227 227 \
#     --image-scale 3 160 160 \
#     --pretrained \
#     --checkpoint /oscar/data/tserre/xyu110/pytorch-output/train/alexnet_size_227_scale_0.08/last.pth.tar \
#     --results-file /oscar/data/tserre/xyu110/pytorch-output/train/scale_image_alexnet_0.08/scale_227_alex_size_$
