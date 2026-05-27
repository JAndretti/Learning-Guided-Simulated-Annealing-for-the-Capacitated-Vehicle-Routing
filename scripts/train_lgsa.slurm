#!/bin/bash
#SBATCH --job-name=LGSA_train
#SBATCH -A aik@v100           # aik@a100 / aik@v100 
#SBATCH -C v100            # a100 / v100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1          # single task — multiprocessing handles the 4 GPUs
#SBATCH --gres=gpu:4                 # 4 GPUs per node
#SBATCH --cpus-per-task=40           # 10 CPUs per GPU × 4 on V100
#SBATCH --hint=nomultithread
#SBATCH --time=20:00:00              # Max time (HH:MM:SS) max 20h a100 / 100h pour v100
#SBATCH --output=logs/lgsa_%j.out
#SBATCH --error=logs/lgsa_%j.out

# 1. Clean environment and load modules
module purge
# module load arch/a100                # MUST have for a100 else comment
module load pytorch-gpu/py3/2.8.0  # MUST match the version you used to install

set -x  # affiche les commandes dans les logs

# Force WandB to save locally instead of hanging
export WANDB_MODE=offline

# Force Python to print instantly to the .out file
export PYTHONUNBUFFERED=1

# Force tqdm to only print an update every 30 seconds
export TQDM_MININTERVAL=30

export UV_CACHE_DIR=$WORK/.uv_cache
export UV_LINK_MODE=copy

# 3. Run your code!
cd $WORK/Learning-Guided-Simulated-Annealing-for-the-Capacitated-Vehicle-Routing
rm -rf wandb/LG-SA/wandb
source .venv/bin/activate
srun python launch_HP_sweep.py