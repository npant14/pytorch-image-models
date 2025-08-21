#!/bin/bash

# Get parameters with defaults
model=${1:-"resmax_v2"}
ip_bands=${2:-1}
cl_lambda=${3:-0}
cls_input_size=${4:-0}
gpus=${5:-8}
batch_size=${6:-128}
bypass=${7:-False}
time_limit=${8:-"120:00:00"}  # Default: 120 hours, or "local"
lr=${9:-0.01}  # Default learning rate
partition=${10:-"gpu"}  # "gpu", "gpu-he", "gracehopper"
scale=${11:-1.0}  # or 0.08
input_size=${12:-322} # or 227

# Add bypass suffix if True
if [ "$bypass" = "True" ]; then
    bypass_str="_bypass"
else
    bypass_str=""
fi

# Create job name
job_name="${model}_ip${ip_bands}_cl${cl_lambda}_cls${cls_input_size}_gpu${gpus}_b${batch_size}${bypass_str}_size_${input_size}_scale_${scale}"

mkdir -p job_scripts

# Create temporary job script in the subdirectory
temp_script="job_scripts/job_${job_name}.sh"
cp bash_scripts/submit_template.sh $temp_script

cpu_per_gpu=1
mem_per_gpu=8

if [ "${partition}" = "gpu-he" ]; then
    cpu_per_gpu=4
    mem_per_gpu=16
fi

cpus=$((gpus * cpu_per_gpu))
mem=$((gpus * mem_per_gpu))
mem="${mem}GB"

# Replace placeholders with actual values
sed -i "s|JOB_NAME|${job_name}|g" $temp_script
sed -i "s|MODEL_NAME|${model}|g" $temp_script
sed -i "s|LR_VALUE|${lr}|g" $temp_script
sed -i "s|IP_BANDS_VALUE|${ip_bands}|g" $temp_script
sed -i "s|CL_LAMBDA_VALUE|${cl_lambda}|g" $temp_script
sed -i "s|CLS_INPUT_SIZE|${cls_input_size}|g" $temp_script
sed -i "s|GPU_COUNT|${gpus}|g" $temp_script
sed -i "s|CPU_COUNT|${cpus}|g" $temp_script
sed -i "s|WORKERS_VALUE|${cpus}|g" $temp_script
sed -i "s|BATCH_SIZE_VALUE|${batch_size}|g" $temp_script
sed -i "s|BYPASS_VALUE|${bypass}|g" $temp_script
sed -i "s|BYPASS_STR_VALUE|${bypass_str}|g" $temp_script
sed -i "s|PARTITION_VALUE|${partition}|g" $temp_script
sed -i "s|MEM_VALUE|${mem}|g" $temp_script
sed -i "s|TIME_LIMIT|${time_limit}|g" $temp_script
sed -i "s|IMAGE_SCALE_VALUE|${scale}|g" $temp_script
sed -i "s|INPUT_SIZE_VALUE|${input_size}|g" $temp_script

# use carney-tserre-condo for gpu jobs
if [ "${partition}" = "gpu" ]; then
    sed -i "/#SBATCH --partition/a #SBATCH --account=carney-tserre-condo" $temp_script
fi
if [ "${partition}" = "gracehopper" ]; then
    sed -i "/#SBATCH --partition/a #SBATCH -account=ccv-gh200-gcondo" $temp_script
fi

if [ $time_limit = "local" ]; then
    # Run the job locally
    bash $temp_script
    exit
fi

# Submit the job
sbatch $temp_script

echo "Submitted job: $job_name with ${gpus} gpu(s) on ${partition}, with ${cpus} cores and ${mem} memory. Time limit: ${time_limit}."
