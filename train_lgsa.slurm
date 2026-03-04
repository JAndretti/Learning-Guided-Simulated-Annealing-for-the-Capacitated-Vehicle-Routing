#!/bin/bash
#SBATCH --job-name=LGSA_train
#SBATCH --account=aik@v100  
#SBATCH --constraint=v100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1          # Number of GPUs you want
#SBATCH --gres=gpu:1                 # Must match ntasks-per-node
#SBATCH --cpus-per-task=10           # 10 CPUs per V100 GPU
#SBATCH --hint=nomultithread
#SBATCH --time=03:00:00              # Max time (HH:MM:SS)
#SBATCH --output=logs/lgsa_%j.out
#SBATCH --error=logs/lgsa_%j.err

# 1. Clean environment and load modules
module purge
module load pytorch-gpu/py3/2.8.0  # MUST match the version you used to install

# 3. Run your code!
cd $WORK/Learning-Guided-Simulated-Annealing-for-the-Capacitated-Vehicle-Routing
srun uv run python launch_HP_sweep.py