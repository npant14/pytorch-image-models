#!/bin/bash
#SBATCH --job-name=pasupathy_new_array
#SBATCH --partition=gpu-he
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
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
models=("CHRESMAX_V3_2" "ALEXNET-AUG" "RESNET18-AUG" "RESNET50" "ALEXNET-NO_AUG" "RESNET18-NO_AUG" "CHRESMAX_V3_2_ABS")

# Get the model for this array task
model=${models[$SLURM_ARRAY_TASK_ID]}

echo "Running ENHANCED Pasupathy evaluation for model: $model"
echo "Array task ID: $SLURM_ARRAY_TASK_ID"

# Configuration options - modify these as needed:
# --use-new: Use enhanced analysis (required)
# --no-neuron-analysis: Disable detailed neuron analysis (faster, less detailed)
# Add --no-neuron-analysis flag below if you want faster analysis without detailed neuron statistics

# Run the evaluation for this specific model using enhanced analysis
python evaluate_hmax_pasupathy.py --model $model --use-new

echo "Evaluation completed for model: $model"
