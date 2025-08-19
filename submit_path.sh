#!/bin/bash
#SBATCH --job-name=pasupathy_eval
#SBATCH --partition=gpu-he
#SBATCH --account=carney-tserre-condo
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=48:00:00
#SBATCH --output=/oscar/data/tserre/xyu110/pytorch-output/%x_%A_%a.out
#SBATCH --error=/oscar/data/tserre/xyu110/pytorch-output/%x_%A_%a.err
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

python evaluate_hmax_pasupathy1.py
