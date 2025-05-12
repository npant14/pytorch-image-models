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


model=cornet_s

IMG_SIZE=322
V2_iterations=1
V4_iterations=1
IT_iterations=1
for imgscale in 160 192 227 271 322 381 454
do
        sh distributed_val.sh 1 validate.py \
            --data-dir /gpfs/data/tserre/npant1/ILSVRC/ \
            --model $model \
            -b 128 \
            --model-kwargs V2_iterations=$V2_iterations V4_iterations=$V4_iterations IT_iterations=$IT_iterations \
            --image-scale 3 $imgscale $imgscale \
            --input-size 3 $IMG_SIZE $IMG_SIZE \
            --checkpoint /users/irodri15/data/irodri15/Hmax/pytorch-image-models/output/train/4_25/${model}_debug2_322_${V2_iterations}_${V4_iterations}_${IT_iterations}/last.pth.tar \
            --results-file /users/irodri15/data/irodri15/Hmax/pytorch-image-models/output/validation/scale_${imgscale}_${model}_debug2_322_${V2_iterations}_${V4_iterations}_${IT_iterations}.txt
        
done