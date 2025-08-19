#!/bin/bash
#SBATCH --job-name=pasupathy_array
#SBATCH --partition=gpu
#SBATCH --account=carney-tserre-condo
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --array=0-6
#SBATCH --output=/users/xyu110/pytorch-image-models/%x_%A_%a.out
#SBATCH --error=/users/xyu110/pytorch-image-models/%x_%A_%a.err
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

# Define array of models
models=("RESNET50" "ALEXNET-NO_AUG" "ALEXNET-AUG" "RESNET18-NO_AUG" "RESNET18-AUG" "CHRESMAX_V3_2_ABS" "CHRESMAX_V3_2")

# Get the model for this array task
model=${models[$SLURM_ARRAY_TASK_ID]}

echo "Running evaluation for model: $model"
echo "Array task ID: $SLURM_ARRAY_TASK_ID"

# Run the evaluation for this specific model
python evaluate_hmax_pasupathy.py --model $model
