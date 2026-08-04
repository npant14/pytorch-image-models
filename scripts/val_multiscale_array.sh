#!/bin/bash
#SBATCH --time=160:00:00
#SBATCH -p gpu --gres=gpu:2
#SBATCH -n 2
#SBATCH -N 1
#SBATCH --mem=40GB
#SBATCH -o scale_invariance_analysis_%A_%a.out
#SBATCH -e scale_invariance_analysis_%A_%a.err
#SBATCH --account=carney-tserre-condo
#SBATCH -J scale_inv_analysis
#SBATCH --array=0-20

module load anaconda/2023.09-0-7nso27y
module load python/3.9.16s-x3wdtvt
module load cuda

source /users/irodri15/data/irodri15/Hmax/hmax_pytorch/venv/bin/activate

# Get the folder path for this array job
FOLDERS=($(ls -d output/5_25/*/))
FOLDER_PATH=${FOLDERS[$SLURM_ARRAY_TASK_ID]}
FOLDER_NAME=$(basename "$FOLDER_PATH")

# Skip if no folder found for this array index
if [ -z "$FOLDER_PATH" ]; then
    echo "No folder found for array index $SLURM_ARRAY_TASK_ID"
    exit 0
fi

# Create results directory
RESULTS_DIR="output/scale_invariance_results"
mkdir -p "$RESULTS_DIR"

# Extract parameters from args.yaml using Python
ARGS_YAML="$FOLDER_PATH/args.yaml"
if [ ! -f "$ARGS_YAML" ]; then
    echo "args.yaml not found in $FOLDER_PATH"
    exit 1
fi

read -r TEACHER_IP STUDENT_IP MODEL_TYPE SIZE <<< $(python3 -c "
import yaml
with open('$ARGS_YAML') as f:
    args = yaml.safe_load(f)
print(args['model_kwargs']['ip_scale_bands'], args['model_kwargs']['ip_scale_bands_student'], args['model'], args['model_kwargs']['classifier_input_size'])
")

# Create a results file for this configuration
RESULTS_FILE="$RESULTS_DIR/${FOLDER_NAME}_scale_results.csv"
echo "scale,accuracy" > "$RESULTS_FILE"

# Run validation for different scales
for imgscale in 160 192 227 322 381 454 
do
    echo "Processing scale $imgscale for $FOLDER_NAME"
    
    # Find the latest checkpoint in the folder
    CHECKPOINT=$(find "$FOLDER_PATH" -name "checkpoint-*.pth.tar" | sort -V | tail -n 1)
    
    if [ -z "$CHECKPOINT" ]; then
        echo "No checkpoint found in $FOLDER_PATH"
        continue
    fi
    
    # Run validation
    python validate_contrastive.py \
        --data-dir /gpfs/data/tserre/npant1/ILSVRC/ \
        --model ${MODEL_TYPE} \
        --model-kwargs ip_scale_bands=${TEACHER_IP} ip_scale_bands_student=${STUDENT_IP} classifier_input_size=${SIZE} bypass=True contrastive_loss=True \
        -b 64 \
        --image-scale 3 $imgscale $imgscale \
        --input-size 3 322 322 \
        --pretrained \
        --checkpoint "$CHECKPOINT" \
        --results-file "$RESULTS_DIR/temp_${imgscale}.txt"
    
    # Extract accuracy and append to results file
    ACCURACY=$(grep "Accuracy" "$RESULTS_DIR/temp_${imgscale}.txt" | awk '{print $2}')
    echo "$imgscale,$ACCURACY" >> "$RESULTS_FILE"
    
    # Clean up temporary file
    rm "$RESULTS_DIR/temp_${imgscale}.txt"
done

echo "Completed processing $FOLDER_NAME" 