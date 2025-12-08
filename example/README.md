# Example Usage

This directory contains an example setup to demonstrate the LGSA approach for CVRP using a trained model.

## Directory Structure

- **`model/`**: Contains the trained model checkpoints used for the inference example.
- **`plots/`**: Destination folder where the visualization results of the experiments will be saved.
- **`example.py`**: The main script to run the evaluation on a set of test problems.

## Running the Example

To run the experiment on 10 sample problems and generate visualizations, execute the following command:

```bash
uv run example.py
```

The script will load the model from the `model/` directory, solve the CVRP instances, and save the resulting route plots to the `plots/` directory.