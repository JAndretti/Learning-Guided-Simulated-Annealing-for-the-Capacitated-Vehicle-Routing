#!/bin/bash

# Define the project path
PROJECT_DIR="$WORK/Learning-Guided-Simulated-Annealing-for-the-Capacitated-Vehicle-Routing"

# 1. Navigate to the project directory
if [ -d "$PROJECT_DIR" ]; then
    cd "$PROJECT_DIR" || exit
else
    echo "Error: Directory $PROJECT_DIR does not exist."
    exit 1
fi

# 2. Activate the virtual environment
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
else
    echo "Error: Virtual environment not found in $PROJECT_DIR"
    exit 1
fi

# 3. Login to Weights & Biases
if [ -f "src/key.txt" ]; then
    wandb login $(cat src/key.txt)
else
    echo "Error: src/key.txt not found. Cannot login to wandb."
    exit 1
fi

# 4. Sync offline runs
echo "Syncing offline wandb runs..."
wandb sync wandb/LG-SA/wandb/offline-*

echo "Done!"