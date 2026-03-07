#!/bin/bash

# Define the source and the base local destination
REMOTE_PATH="JeanZay:/lustre/fswork/projects/rech/aik/uwa53wm/Learning-Guided-Simulated-Annealing-for-the-Capacitated-Vehicle-Routing/wandb/LG-SA/"
LOCAL_BASE="/home/jules/Documents/LGSA-CVRP/wandb/LG-SA"

# Ensure the base directory exists locally
mkdir -p "$LOCAL_BASE"

echo "========================================"
echo " Jean Zay WandB Sync Utility"
echo "========================================"

# Interactive loop for naming and safety
while true; do
    echo ""
    # 1. Ask for the target directory name upfront
    read -p "Enter the target folder name (Press Enter for 'Critic'): " TARGET_DIR
    
    # If the user leaves it blank, default to "Critic"
    TARGET_DIR=${TARGET_DIR:-Critic} 

    REMOTE_PATH_WITH_TARGET="$REMOTE_PATH$TARGET_DIR/"
    
    FINAL_DEST="$LOCAL_BASE/$TARGET_DIR"

    # 2. Check if this specific folder already exists
    if [ -d "$FINAL_DEST" ]; then
        echo "⚠️  WARNING: The folder '$TARGET_DIR' already exists locally."
        read -p "Do you want to (S)ync/Update it, (R)ename, or (C)ancel? [S/r/c]: " choice
        
        # Default to 'S' if the user just presses Enter
        choice=${choice:-S} 

        case "$choice" in
            [Ss]* )
                echo "-> Proceeding to sync and update the existing folder."
                break
                ;;
            [Rr]* )
                echo "-> Let's pick a different name."
                # The loop restarts, prompting for the name again
                ;;
            [Cc]* )
                echo "-> Sync cancelled by user. Exiting."
                exit 0
                ;;
            * )
                echo "-> Invalid input. Let's try again."
                ;;
        esac
    else
        echo "-> Target folder '$TARGET_DIR' is clear. Proceeding..."
        break
    fi
done

echo ""
echo "🚀 Starting secure transfer to: $FINAL_DEST"
# The trailing slash on REMOTE_PATH ensures we put the contents inside FINAL_DEST
rsync -avP "$REMOTE_PATH_WITH_TARGET" "$FINAL_DEST/"

echo ""
echo "✅ Transfer complete!"