#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH -p gpu-he --gres=gpu:4
#SBATCH -n 4
#SBATCH -N 1
#SBATCH --mem=80GB
#SBATCH -J hmax_timm
#SBATCH -o /users/npant1/pytorch-image-models/hmax_out.txt
#SBATCH -e /users/npant1/pytorch-image-models/hmax_err.txt
#SBATCH --mail-user=nishka_pant@brown.edu
#SBATCH --mail-type=END,FAIL


cd /users/npant1/hmax_pytorch_new

# module load anaconda/2022.05
# module load anaconda/3-5.2.0
module load anaconda/2023.09-0-7nso27y
# module load python/3.9.12
module load python/3.9.16s-x3wdtvt
# module load opencv/3.2.0
module load cuda
source /users/npant1/anaconda3/bin/activate /users/npant1/anaconda3/envs/hmax


./users/npant1/pytorch-image-models/run_model.sh