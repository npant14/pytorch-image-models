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


source  /users/irodri15/data/irodri15/Hmax/hmax_pytorch/venv/bin/activate

for imgscale in 160 192 227 271 322 
do
    for ip_band in 7
    do  
        # Set size based on ip_band
        if [ $ip_band -eq 1 ]; then
            size=9216
        else
            size=9216
        fi

        sh distributed_val.sh 1 validate.py \
            --data-dir /gpfs/data/tserre/npant1/ILSVRC/ \
            --model chalexmax_v3_3\
            --model-kwargs ip_scale_bands=${ip_band} classifier_input_size=${size} \
            -b 128 \
            --image-scale 3 $imgscale $imgscale \
            --input-size 3 322 322 \
            --checkpoint /users/irodri15/data/irodri15/Hmax/pytorch-image-models/output/2_25/debug4_resize2_{chalexmax_v3_3}_cl_{1}_ip_{$ip_band}_322_{9216}/model_best.pth.tar \
            --results-file /users/irodri15/data/irodri15/Hmax/pytorch-image-models/output/2_25/scale_${imgscale}_chalexmax_v3_3_bypass_ip_${ip_band}.txt
        wait

    done
done