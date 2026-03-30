#!/bin/bash
#SBATCH --job-name=LGSA_train
#SBATCH --account=aik@v100           # aik@a100 / aik@v100 
#SBATCH --constraint=v100            # a100 / v100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1          # Nombre de GPU
#SBATCH --gres=gpu:1                 # Doit correspondre à ntasks-per-node
#SBATCH --cpus-per-task=10           # 8 CPUs par GPU sur A100 / 10 sur V100
#SBATCH --hint=nomultithread
#SBATCH --time=6:00:00              # Max time (HH:MM:SS) max 20h a100 / 100h pour v100
#SBATCH --output=logs/lgsa_%j.out
#SBATCH --error=logs/lgsa_%j.out

# 1. Clean environment and load modules
module purge
# module load arch/a100                # OBLIGATOIRE pour a100 sinon commenter la ligne
module load pytorch-gpu/py3/2.8.0  # MUST match the version you used to install

# Force WandB to save locally instead of hanging
export WANDB_MODE=offline

# Force Python to print instantly to the .out file
export PYTHONUNBUFFERED=1

# Force tqdm to only print an update every 30 seconds
export TQDM_MININTERVAL=30

# 3. Run your code!
cd $WORK/Learning-Guided-Simulated-Annealing-for-the-Capacitated-Vehicle-Routing
rm -rf wandb/LG-SA/wandb
srun uv run python launch_HP_sweep.py