#!/bin/bash
#SBATCH --time=160:00:00
#SBATCH -p gpu --gres=gpu:2
#SBATCH -n 4
#SBATCH -N 1
#SBATCH --mem=80GB
#SBATCH -o alexmax_cl_0_ip_5.out
#SBATCH -e alexmax_cl_0_ip_5.err
#SBATCH --account=carney-tserre-condo
#SBATCH -J alexmax_cl_0_ip_5_2


module load anaconda/2023.09-0-7nso27y
module load python/3.9.16s-x3wdtvt
module load cuda

source  /users/irodri15/data/irodri15/Hmax/hmax_pytorch/venv/bin/activate



MODE='chresmax_v3_3_abs'
for imgscale in 160 192 227 272 322 384 448 512
do
    for ip_band in 3
    do  
        # Set size based on ip_band
        if [ $ip_band -eq 1 ]; then
            size=4096
        else
            size=18432
        fi
        sh distributed_val.sh 2 validate.py \
            --data-dir /gpfs/data/tserre/npant1/ILSVRC/ \
            --model ${MODE} \
            --model-kwargs ip_scale_bands=${ip_band} ip_scale_bands_student=3 classifier_input_size=${size} bypass=True\
            -b 64 \
            --image-scale 3 $imgscale $imgscale \
            --input-size 3 322 322 \
            --pretrained \
            --checkpoint /users/irodri15/data/irodri15/Hmax/pytorch-image-models/output/train/6_25/debug10.3_balanced_scale_{chresmax_v3_3_abs}_bypass_cl_{0.1}_ip_{3}_322_{18432}/model_best.pth.tar \
            --results-file output/validation/scale_${imgscale}_{$MODE}_ip_${ip_band}.txt
        wait

    done
done