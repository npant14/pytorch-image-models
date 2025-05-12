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


# module load anaconda/2023.09-0-7nso27y
# module load python/3.9.16s-x3wdtvt
# module load cuda

# source  /users/irodri15/data/irodri15/Hmax/hmax_pytorch/venv/bin/activate

MODEL=chalexmax_bypass_dl_v2
SCALE_BANDS=5
ALPHA=10
CL_LAMBDA=0.1
echo "Starting experiment"
sh distributed_stu_dl.sh 1 train_skeleton_dl.py \
    --data-dir /oscar/data/tserre/npant1/ILSVRC/ \
    --dataset torch/imagenet \
    --model $MODEL \
    --model-kwargs ip_scale_bands=$SCALE_BANDS classifier_input_size=9216 \
    --scale-bands $SCALE_BANDS \
    --opt sgd \
    -b 256 \
    --epochs 90 \
    --cl-lambda $CL_LAMBDA \
    --alpha $ALPHA \
    --lr 1e-3 \
    --weight-decay 5e-4 \
    --sched step \
    --momentum 0.9 \
    --lr-cycle-decay 0.1 \
    --decay-epochs 30 \
    --warmup-epochs 0 \
    --hflip 0.5\
    --train-crop-mode rrc\
    --scale 1.0 1.0 \
    --input-size 3 322 322\
    --experiment gracehopper_debug2_dl_${MODEL}_${SCALE_BANDS}_322_9216_alpha_${ALPHA}_cl_lambda_${CL_LAMBDA} \
    --output /users/irodri15/data/irodri15/Hmax/pytorch-image-models/output/train/4_25/\
    
 
