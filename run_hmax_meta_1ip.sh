#!/bin/bash

# Array of cl-lambda values to iterate over
cl_lambda_values=(0 0.1 1 10)

for cl_lambda in "${cl_lambda_values[@]}"; do
    # Update the job name and output/error filenames based on cl_lambda
    job_name="results/hmax_bypass_cl_${cl_lambda}_ip_1"
    output_file="results/hmax_bypass_cl_${cl_lambda}_ip_1.out"
    error_file="results/hmax_bypass_cl_${cl_lambda}_ip_1.err"
    
    # Create the job script for each cl-lambda value
    cat << EOF > "job_script_ip_1_${cl_lambda}.sh"
#!/bin/bash
#SBATCH --time=99:00:00
#SBATCH -p gpu --gres=gpu:2
#SBATCH -n 4
#SBATCH -N 1
#SBATCH --mem=80GB
#SBATCH -o ${output_file}
#SBATCH -e ${error_file}
#SBATCH --account=carney-tserre-condo
#SBATCH -J ${job_name}

module load anaconda/2023.09-0-7nso27y
module load python/3.9.16s-x3wdtvt
module load cuda

source  /users/irodri15/data/irodri15/Hmax/hmax_pytorch/venv/bin/activate

sh distributed_train.sh 2 train_skeleton.py \
    --data-dir /gpfs/data/tserre/npant1/ILSVRC/ \
    --dataset torch/imagenet \
    --model chmax \
    --model-kwargs ip_scale_bands=1 classifier_input_size=4096 hmax_type='bypass' \
    --cl-lambda ${cl_lambda} \
    --opt sgd \
    -b 128 \
    --epochs 90 \
    --lr 1e-2 \
    --weight-decay 5e-4 \
    --sched step \
    --momentum 0.9 \
    --lr-cycle-decay 0.1 \
    --decay-epochs 30 \
    --warmup-epochs 0 \
    --hflip 0.5 \
    --train-crop-mode rrc \
    --input-size 3 227 227 \
    --experiment bypass_cl_${cl_lambda}_ip_1 \
    --output /users/irodri15/data/irodri15/Hmax/pytorch-image-models/output/train
EOF

    # Submit the job script
    sbatch "job_script_ip_1_${cl_lambda}.sh"

    # Optionally, remove the job script after submission to clean up
    rm "job_script_ip_1_${cl_lambda}.sh"
done