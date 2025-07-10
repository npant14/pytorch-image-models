#!/bin/bash
#SBATCH --job-name=korean_eval
#SBATCH --partition=gpu
#SBATCH --account=carney-tserre-condo
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=/oscar/data/tserre/xyu110/pytorch-output/korean/slurm_logs/%x_%A_%a.out
#SBATCH --error=/oscar/data/tserre/xyu110/pytorch-output/korean/slurm_logs/%x_%A_%a.err
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

MODEL_TO_RUN="chresmax_abs_bypass_only"
JSON_FILE="/users/xyu110/pytorch-image-models/korean/models_and_layers.json"

cd /users/xyu110/pytorch-image-models
mkdir -p /oscar/data/tserre/xyu110/pytorch-output/korean/slurm_logs

# wait 10 seconds
sleep 10

if ! command -v jq &> /dev/null; then
    echo "Error: jq is not installed or not in your PATH. Please install it to parse the JSON config."
    exit 1
fi

# Use jq to get the number of layers for the specified model
# The result is the length of the JSON array.
NUM_LAYERS=$(jq ".[\"$MODEL_TO_RUN\"] | length" "$JSON_FILE")
ARRAY_END=$((NUM_LAYERS - 1))

# This resubmits the script with the --array option, preventing an infinite loop.
if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
    sbatch --array=0-$ARRAY_END $0
    echo "Submitted job array."
    exit
fi

LAYER_NAME=$(jq -r ".[\"$MODEL_TO_RUN\"][$SLURM_ARRAY_TASK_ID]" "$JSON_FILE")

echo "--------------------------------------------------------"
echo "SLURM JOB ID: $SLURM_JOB_ID"
echo "SLURM ARRAY TASK ID: $SLURM_ARRAY_TASK_ID"
echo "Processing Model: $MODEL_TO_RUN"
echo "Processing Layer: $LAYER_NAME"
echo "--------------------------------------------------------"

# The python script does not need to change at all.
python korean.py --model_name "$MODEL_TO_RUN" --layer_name "$LAYER_NAME"

echo "Job finished."