#!/bin/bash
#SBATCH --job-name=korean_array_eval
#SBATCH --partition=gpu
#SBATCH --account=carney-tserre-condo
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --array=0-3
#SBATCH --output=./%x_%A_%a.out
#SBATCH --error=./%x_%A_%a.err
#SBATCH --mail-user=xizheng_yu@brown.edu
#SBATCH --mail-type=END,FAIL

# Check if modules are loaded
if ml 2>&1 | grep -q "No modules loaded"; then
    module load miniconda3/23.11.0s
    source /oscar/runtime/software/external/miniconda3/23.11.0/etc/profile.d/conda.sh
    conda activate env_default
else
    echo "Modules already loaded, skipping conda activation"
fi

which python

cd /users/xyu110/pytorch-image-models
mkdir -p /oscar/data/tserre/xyu110/pytorch-output/korean/slurm_logs

# wait 10 seconds
sleep 10

echo "--------------------------------------------------------"
echo "SLURM JOB ID: $SLURM_JOB_ID"
echo "Running array job for multiple models"
echo "--------------------------------------------------------"

# Define array of models
models=("chresmax_v3_2" "chresmax_v3_2_abs" "resnet18" "alexnet")

# Get the model for this array task
model_name=${models[$SLURM_ARRAY_TASK_ID]}

echo "Running model: $model_name (Array Task ID: $SLURM_ARRAY_TASK_ID)"

# Run the specific command for this model
python korean_imagenet.py --model_name $model_name --run_all_layers

echo "Job finished."
