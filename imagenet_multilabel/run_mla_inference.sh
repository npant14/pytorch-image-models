#!/bin/bash
#SBATCH --job-name=mla_inference
#SBATCH --partition=gpu-he
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=8:00:00
#SBATCH --array=0-3
#SBATCH --output=./%x_%A_%a.out
#SBATCH --error=./%x_%A_%a.err
#SBATCH --mail-user=xizheng_yu@brown.edu
#SBATCH --mail-type=END,FAIL

# Define models, checkpoints, and input sizes for each array task
case $SLURM_ARRAY_TASK_ID in
    0)
        MODEL="hmax_v3_adj"
        CHECKPOINT="/oscar/data/tserre/xyu110/pytorch-output/train/0/final_versions/ip_3_hmax_v3_adj_gpu_8_cl_0.1_ip_3_322_322_18432_c1[_6,3,1_]_bypass/model_best.pth.tar"
        INPUT_SIZE=322
        ;;
    1)
        MODEL="alexnet_timm"
        CHECKPOINT=""
        INPUT_SIZE=224
        ;;
    2)
        MODEL="resnet18_timm"
        CHECKPOINT=""
        INPUT_SIZE=224
        ;;
    3)
        MODEL="vit_base"
        CHECKPOINT=""
        INPUT_SIZE=224
        ;;
esac

echo "Running evaluation for model: $MODEL"
echo "Checkpoint: $CHECKPOINT"
echo "Input size: $INPUT_SIZE"

# Run the evaluation with multiscale
if [[ -n "$CHECKPOINT" ]]; then
    python imagenet_multilabel/mla_hmax_inference.py \
        --model $MODEL \
        --checkpoint $CHECKPOINT \
        --input-size $INPUT_SIZE \
        --multiscale
else
    python imagenet_multilabel/mla_hmax_inference.py \
        --model $MODEL \
        --input-size $INPUT_SIZE \
        --multiscale
fi

# python imagenet_multilabel/mla_hmax_inference.py --model vit_base --input-size 224 --multiscale
