#!/bin/bash
#SBATCH -N 1
#SBATCH -c 6
#SBATCH --time 04:00:00
#SBATCH --mem-per-cpu 3G
#SBATCH --job-name tunnel
#SBATCH --partition=gpu
#SBATCH --account=carney-tserre-condo
#SBATCH --gres=gpu:1
#SBATCH --output jupyter-log.txt
## get tunneling info
XDG_RUNTIME_DIR=""
ipnport=$(shuf -i8000-9999 -n1)
ipnip=$(hostname -i)
## print tunneling instructions to jupyter-log-{jobid}.txt
echo -e "
    Copy/Paste this in your local terminal to ssh tunnel with remote
    -----------------------------------------------------------------
    ssh -N -L $ipnport:$ipnip:$ipnport $USER@ssh.ccv.brown.edu
    -----------------------------------------------------------------
    Then open a browser on your local machine to the following address
    ------------------------------------------------------------------
    localhost:$ipnport  (prefix w/ https:// if using password)
    ------------------------------------------------------------------
    "
## start an ipcluster instance and launch jupyter server
module purge
module load miniconda3/23.11.0s-odstpk5 
source /oscar/runtime/software/external/miniconda3/23.11.0/etc/profile.d/conda.sh
conda activate env_default
# jupyter-notebook --no-browser --port=$ipnport --ip=$ipnip
jupyter lab --no-browser --port=$ipnport --ip=$ipnip --ServerApp.token='apple123'
