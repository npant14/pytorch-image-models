#!/bin/bash
#SBATCH --time=3:00:00
#SBATCH -p gpu --gres=gpu:2
#SBATCH -n 4
#SBATCH -N 1
#SBATCH --mem=60GB
#SBATCH -J hmax_bypass_validation
#SBATCH -o /oscar/data/tserre/xyu110/pytorch-output/train/hmax_bypass_validation.out
#SBATCH -e /oscar/data/tserre/xyu110/pytorch-output/train/hmax_bypass_validation.err
#SBATCH --account=carney-tserre-condo

#SBATCH --mail-user=xizheng_yu@brown.edu
#SBATCH --mail-type=END,FAIL


module load miniconda3/23.11.0s
source /oscar/runtime/software/external/miniconda3/23.11.0/etc/profile.d/conda.sh
conda activate env_default

cd /users/xyu110/pytorch-image-models

for imgscale in 160 192 227
do
    for ip_band in 1 2
    do  
        # Set size based on ip_band
        if [ $ip_band -eq 1 ]; then
            size=4096
        else
            size=6400
        fi

        sh distributed_val.sh 2 validate.py \
            --data-dir /gpfs/data/tserre/npant1/ILSVRC/ \
            --model chmax \
            --model-kwargs ip_scale_bands=${ip_band} classifier_input_size=${size} hmax_type='bypass'\
            -b 128 \
            --image-scale 3 $imgscale $imgscale \
            --input-size 3 227 227 \
            --pretrained \
            --checkpoint /oscar/data/tserre/xyu110/pytorch-output/train/hmax_ip_${ip_band}/last.pth.tar \
            --results-file /oscar/data/tserre/xyu110/pytorch-output/train/hmax_bypass_validation/scale_${imgscale}_hmax_bypass_ip_${ip_band}.txt
        wait

    done
done