#!/bin/bash
#SBATCH --job-name=mla_inference
#SBATCH --partition=gpu-he
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=8:00:00
#SBATCH --array=0-4
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
        MODEL="alexnet_aug"
        CHECKPOINT="/oscar/data/tserre/xyu110/pytorch-output/train/0/baseline_w_aug/ip_0_alexnet_gpu_2_cl_0_ip_3_227_227_0_c1[_6,3,1_]_scale_0.08/model_best.pth.tar"
        INPUT_SIZE=227
        ;;
    2)
        MODEL="resnet18_aug"
        CHECKPOINT="/oscar/data/tserre/xyu110/pytorch-output/train/0/baseline_w_aug/ip_0_resnet18_gpu_8_cl_0_ip_3_227_227_512_c1[_6,3,1_]_scale_0.08/model_best.pth.tar"
        INPUT_SIZE=227
        ;;
    3)
        MODEL="alexnet_wo_aug"
        CHECKPOINT="/oscar/data/tserre/xyu110/pytorch-output/train/0/baseline_wo_aug/ip_0_alexnet_gpu_2_cl_0_ip_3_227_227_0_c1[_6,3,1_]/model_best.pth.tar"
        INPUT_SIZE=227
        ;;
    4)
        MODEL="resnet18_wo_aug"
        CHECKPOINT="/oscar/data/tserre/xyu110/pytorch-output/train/0/baseline_wo_aug/ip_0_resnet18_gpu_8_cl_0_ip_3_227_227_512_c1[_6,3,1_]/model_best.pth.tar"
        INPUT_SIZE=227
        ;;
esac

echo "Running evaluation for model: $MODEL"
echo "Checkpoint: $CHECKPOINT"
echo "Input size: $INPUT_SIZE"

# Run the evaluation with multiscale
python imagenet_multilabel/mla_hmax_inference.py \
    --model $MODEL \
    --checkpoint $CHECKPOINT \
    --input-size $INPUT_SIZE \
    --multiscale
