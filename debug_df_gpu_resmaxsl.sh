#!/bin/bash
#SBATCH --time=160:00:00
#SBATCH -p gpu --gres=gpu:8
#SBATCH -n 64
#SBATCH -N 1
#SBATCH --mem=0GB
#SBATCH -o gracehopper_resmaxsl_debug3.out
#SBATCH -e gracehopper_resmaxsl_debug3.err
#SBATCH --account=carney-tserre-condo
#SBATCH -J gracehopper_resmaxsl_debug3

# module load python/3.9.16s-x3wdtvt
# module load cuda

# source  /users/irodri15/data/irodri15/Hmax/hmax_pytorch/venv/bin/activate
# cd /users/irodri15/data/irodri15/Hmax/pytorch-image-models/

MODEL=chresmax_v3_s
SCALE_BANDS=5
ALPHA=1
BYPASS=True
CL_LAMBDA=1


CLASSIFIER_INPUT_SIZE=13312

echo "Starting experiment"
sh distributed_stu_dl.sh 1 train_skeleton_dl.py \
    --data-dir /oscar/data/tserre/npant1/ILSVRC/ \
    --dataset torch/imagenet_scale \
    --model $MODEL \
    --model-kwargs bypass=$BYPASS contrastive_loss=True ip_scale_bands=$SCALE_BANDS classifier_input_size=$CLASSIFIER_INPUT_SIZE \
    --scale-bands $SCALE_BANDS \
    --opt sgd \
    --cl-lambda $CL_LAMBDA \
    -b 32 \
    --epochs 90 \
    --alpha $ALPHA \
    --lr 1e-2 \
    --weight-decay 5e-4 \
    --sched step \
    --momentum 0.9 \
    --lr-cycle-decay 0.1 \
    --decay-epochs 30 \
    --warmup-epochs 2 \
    --hflip 0.5\
    --train-crop-mode rrc\
    --scale 1.0 1.0 \
    --start-epoch 0\
    --workers 4\
    --initial-checkpoint /users/irodri15/data/irodri15/Hmax/pytorch-image-models/output/train/4_25/gracehopper_debug3_dl_chresmax_v3_s_5_322_13312_alpha_1_bypass_True_contrastive_loss/checkpoint-5.pth.tar \
    --input-size 3 322 322\
    --experiment gpu_debug1_dl_${MODEL}_${SCALE_BANDS}_322_${CLASSIFIER_INPUT_SIZE}_alpha_${ALPHA}_bypass_${BYPASS}_contrastive_loss\
    --output /users/irodri15/data/irodri15/Hmax/pytorch-image-models/output/train/4_25/\
    
 
