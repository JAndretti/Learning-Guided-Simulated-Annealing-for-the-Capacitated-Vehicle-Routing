# Evaluation Directory Overview

This directory is designed to evaluate various models and methods, with results stored in the `res` folder.



## eval.py
This script evaluates models on generated datasets with varying parameters. To run it : 
```bash
uv run evaluation/eval.py --INIT '' --dim '' --FOLDER '' --OUTER_STEPS '' --DATA '' --no-baseline --greedy
```

## visu_exp.py
Visualizes data from a specific experiment.  
- **Setup**: Fill the `MODEL_NAME` variable with the name of the model to use for visualization.

## eval_on_bdd.py (might not be up to date)
This script is used to evaluate a model on 10,000 instances of Uchoa / Queiroga dataset.  
- **Setup**: Fill the `MODEL` variable with the name of the model folder .  
- **Configuration**: The `cfg` variable contains configuration parameters. Avoid modifying these for better comparison across models.

## func.py
Contains utility functions used across different scripts. Examples include initializing problem parameters, setting random seeds, loading models, and plotting solutions.



