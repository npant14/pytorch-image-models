#!/bin/bash

# Get parameters with defaults
model=${1:-"resmax_v2"}
ip_bands=${2:-1}
cl_lambda=${3:-0}
cls_input_size=${4:-9216}
gpus=${5:-2}
batch_size=${6:-128}
bypass=${7:-False}
image_scales=${8:-"322"}  # Default is a single scale, but can now take a list like "160 192 227 270"
time_limit=${9:-"3:00:00"}  # Default: 3 hours or "local"
ckpt_dir=${10:-""}
results_dir=${11:-"/oscar/data/tserre/xyu110/pytorch-output/validation"}

# Process the image_scales parameter
# If it contains spaces, it's a list of scales to loop through
# If not, treat it as a single scale
if [[ $image_scales == *" "* ]]; then
    # It's a list of scales, split by spaces
    scales=($image_scales)
else
    # It's a single scale
    scales=($image_scales)
fi

# Loop through each image scale
for image_scale in "${scales[@]}"; do
    echo "Processing image scale: $image_scale"

    # Add bypass suffix if True
    if [ "$bypass" = "True" ]; then
        bypass_str="_bypass"
    else
        bypass_str=""
    fi

    # Create job name
    val_job_name="val_${model}_ip${ip_bands}_cl${cl_lambda}_cls${cls_input_size}_gpu${gpus}_b${batch_size}_scale${image_scale}${bypass_str}"

    mkdir -p val_scripts
    mkdir -p $results_dir

    # Create temporary job script in the subdirectory
    temp_script="val_scripts/${val_job_name}.sh"
    cp val_template.sh $temp_script

    # Replace placeholders with actual values
    sed -i "s/VAL_JOB_NAME/${val_job_name}/g" $temp_script
    sed -i "s/MODEL_NAME/${model}/g" $temp_script
    sed -i "s/IP_BANDS_VALUE/${ip_bands}/g" $temp_script
    sed -i "s/CL_LAMBDA_VALUE/${cl_lambda}/g" $temp_script
    sed -i "s/CLS_INPUT_SIZE/${cls_input_size}/g" $temp_script
    sed -i "s/GPU_COUNT/${gpus}/g" $temp_script
    sed -i "s/BATCH_SIZE_VALUE/${batch_size}/g" $temp_script
    sed -i "s/BYPASS_VALUE/${bypass}/g" $temp_script
    sed -i "s/BYPASS_STR_VALUE/${bypass_str}/g" $temp_script
    sed -i "s/TIME_LIMIT/${time_limit}/g" $temp_script
    sed -i "s|RESULTS_DIR_VALUE|${results_dir}|g" $temp_script
    sed -i "s/IMAGE_SCALE_VALUE/${image_scale}/g" $temp_script
    sed -i "s|CKPT_DIR_VALUE|${ckpt_dir}|g" $temp_script

    # Run based on the mode
    if [ $time_limit = "local" ]; then
        # Run locally
        chmod +x $temp_script
        echo "Running validation job locally: $val_job_name"
        $temp_script
        wait
        echo "Validation job completed: $val_job_name"
    else
        # Submit as a batch job
        sbatch $temp_script
        echo "Submitted validation job: $val_job_name with time limit: ${time_limit}"
    fi
    
    # If time_limit is not "local", add a small delay between job submissions
    if [ "$time_limit" != "local" ]; then
        sleep 1
    fi
done

echo "All validation jobs for specified image scales have been processed."