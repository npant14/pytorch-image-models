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
ALPHA=0
BYPASS=True
CL_LAMBDA=0.1

MAIN_ROUTE=False 
if [ "$MAIN_ROUTE" = True ]; then
    CLASSIFIER_INPUT_SIZE=13312
else
    CLASSIFIER_INPUT_SIZE=18432
fi

echo "Starting experiment"
sh distributed_stu_dl.sh 8 train_skeleton_dl.py \
    --data-dir /oscar/data/tserre/npant1/ILSVRC/ \
    --dataset torch/imagenet_scale \
    --model $MODEL \
    --model-kwargs bypass=$BYPASS main_route=$MAIN_ROUTE contrastive_loss=True ip_scale_bands=$SCALE_BANDS classifier_input_size=$CLASSIFIER_INPUT_SIZE \
    --scale-bands $SCALE_BANDS \
    --opt sgd \
    --cl-lambda $CL_LAMBDA \
    -b 20 \
    --epochs 90 \
    --alpha $ALPHA \
    --lr 1e-2 \
    --weight-decay 5e-4 \
    --sched step \
    --momentum 0.9 \
    --lr-cycle-decay 0.1 \
    --decay-epochs 30 \
    --warmup-epochs 0 \
    --hflip 0.5\
    --train-crop-mode rrc\
    --scale 1.0 1.0 \
    --start-epoch 0\
    --input-size 3 322 322\
    --experiment gpu_debug3_mainroute_${MAIN_ROUTE}_dl_${MODEL}_${SCALE_BANDS}_322_${CLASSIFIER_INPUT_SIZE}_alpha_${ALPHA}_bypass_${BYPASS}_contrastive_loss\
    --output /users/irodri15/data/irodri15/Hmax/pytorch-image-models/output/train/4_25/\
    
 
