sh distributed_train.sh 4 train.py \
    --data-dir /gpfs/data/tserre/npant1/ILSVRC/ \
    --model hmax_full \
    --opt sgd \
    -b 16 \
    --epochs 90 \
    --lr 1e-2 \
    --weight-decay 1e-4 \
    --sched step \
    --lr-cycle-decay 0.1 \
    --decay-epochs 30 \
    --warmup-epochs 0 \
    --grad-accum-steps 64 \
    --clip-grad 1 \
    --output /gpfs/data/tserre/npant1/pytorch-output/train \
    -j 1 \
    --resume /gpfs/data/tserre/npant1/pytorch-output/train/20240713-123446-hmax_full-224/checkpoint-34.pth.tar
    --resume /gpfs/data/tserre/npant1/pytorch-output/train/20240708-120804-hmax_full-224/checkpoint-16.pth.tar
    --resume /gpfs/data/tserre/npant1/pytorch-output/train/20240626-114151-hmax_full-224/checkpoint-8.pth.tar

## alexnet training recipe
# sh distributed_train.sh 4 train.py \
#     --data-dir /gpfs/data/tserre/npant1/ILSVRC/ \
#     --model alexnet \
#     --opt sgd \
#     -b 1024 \
#     --epochs 90 \
#     --lr 1e-2 \
#     --weight-decay 1e-4 \
#     --sched step \
#     --lr-cycle-decay 0.1 \
#     --decay-epochs 30 \
#     --warmup-epochs 0 \
#     -j 1 \
#     --resume /users/npant1/pytorch-image-models/output/train/20240606-115444-alexnet-224/checkpoint-44.pth.tar


    ## resnet training recipes
    # --epochs 100 \
    # --opt sgd \
    # -b 512 \
    # --lr 8e-3 \
    # --opt-eps 1e-6 \
    # --sched cosine \
    # --weight-decay 0.02 \
    # --hflip 0.5 \
    # --train-crop-mode rrc \
    # --aa rand-m6-n4-mstd1.0-inc1 \
    # --mixup 0.1 \
    # --cutmix 1.0 \
    # --bce-loss \
    # --amp \
    # -j 4 \
    # --resume /users/npant1/pytorch-image-models/output/train/20240511-180955-resnet50-224/last.pth.tar

    # --model-kwargs s1_channels_out=96 \