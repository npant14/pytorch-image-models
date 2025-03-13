#!/bin/bash

# Get parameters with defaults
model=${1:-"resmax_v2"}
ip_bands=${2:-1}
cl_lambda=${3:-0}
cls_input_size=${4:-9216}
gpus=${5:-8}
batch_size=${6:-128}
bypass=${7:-False}
time_limit=${8:-"120:00:00"}  # Default: 120 hours

# Add bypass suffix if True
if [ "$bypass" = "True" ]; then
    bypass_str="_bypass"
else
    bypass_str=""
fi

# Create job name
job_name="${model}_ip${ip_bands}_cl${cl_lambda}_cls${cls_input_size}_gpu${gpus}_b${batch_size}${bypass_str}"

mkdir -p job_scripts

# Create temporary job script in the subdirectory
temp_script="job_scripts/job_${job_name}.sh"
cp submit_template.sh $temp_script

# Replace placeholders with actual values
sed -i "s/JOB_NAME/${job_name}/g" $temp_script
sed -i "s/MODEL_NAME/${model}/g" $temp_script
sed -i "s/IP_BANDS_VALUE/${ip_bands}/g" $temp_script
sed -i "s/CL_LAMBDA_VALUE/${cl_lambda}/g" $temp_script
sed -i "s/CLS_INPUT_SIZE/${cls_input_size}/g" $temp_script
sed -i "s/GPU_COUNT/${gpus}/g" $temp_script
sed -i "s/BATCH_SIZE_VALUE/${batch_size}/g" $temp_script
sed -i "s/BYPASS_VALUE/${bypass}/g" $temp_script
sed -i "s/BYPASS_STR_VALUE/${bypass_str}/g" $temp_script
sed -i "s/TIME_LIMIT/${time_limit}/g" $temp_script

# Submit the job
sbatch $temp_script

echo "Submitted job: $job_name with time limit: ${time_limit}"
