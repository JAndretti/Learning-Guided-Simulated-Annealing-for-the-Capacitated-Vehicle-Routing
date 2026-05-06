import os

import glob2
import numpy as np
import pandas as pd
import torch
import vrplib
import yaml


def _checkpoint_loss(path: str) -> float:
    """Extract the loss value embedded in a checkpoint filename (e.g. actor_0.1234.pt)."""
    name = os.path.basename(path)
    try:
        return float(name.split("_")[-1])
    except ValueError:
        return float("inf")


def find_model(folder: str) -> str:
    """Return the best (lowest-loss) model path in wandb/LGSA/<folder>/models/."""
    paths = glob2.glob(os.path.join("wandb", "LGSA", folder, "models", "*"))
    if not paths:
        raise FileNotFoundError(f"No models found in wandb/LGSA/{folder}/models/")
    return min(paths, key=_checkpoint_loss)


def find_models(folder: str) -> list[str]:
    """Return all model paths in wandb/LGSA/<folder>/models/, sorted by loss ascending."""
    paths = glob2.glob(os.path.join("wandb", "LGSA", folder, "models", "*"))
    if not paths:
        raise FileNotFoundError(f"No models found in wandb/LGSA/{folder}/models/")
    return sorted(paths, key=_checkpoint_loss)


def get_HP_for_model(model_path: str) -> dict:
    """Load and parse HP.yaml for a given model directory."""
    hp_file = os.path.join(model_path, "HP.yaml")
    with open(hp_file, "r") as f:
        content = f.read().replace("!!python/object:HP._HP", "")
    hp_data = yaml.unsafe_load(content)
    return hp_data.config


def init_problem_parameters(model_path: str, cfg: dict) -> dict:
    """Merge model HP.yaml with runtime cfg overrides. cfg values take precedence."""
    HP = get_HP_for_model(model_path)
    HP.update(cfg)
    return HP


def load_vrp_instance(
    filepath: str,
    load_solution: bool = True,
    sol_dir: str | None = None,
) -> dict:
    """
    Load a .vrp file and return normalised tensors plus raw data.

    Args:
        filepath:      Path to the .vrp file.
        load_solution: Whether to attempt loading the matching .sol file.
        sol_dir:       Directory to look for .sol file. If None, looks adjacent
                       to the .vrp file (same directory, same stem).

    Returns dict with keys:
        name         - instance name (stem of filepath)
        raw_coords   - np.ndarray [N, 2]
        raw_demand   - np.ndarray [N]
        capacity     - int/float scalar
        norm_coords  - torch.Tensor [N, 2], min-max scaled to [0, 1]
        n_nodes      - int (includes depot)
        optimal_cost - float (nan if unavailable or load_solution=False)
    """
    instance_name = os.path.basename(filepath).replace(".vrp", "")
    vrp_data = vrplib.read_instance(filepath)

    coords = vrp_data["node_coord"]
    demand = vrp_data["demand"]
    capacity = vrp_data["capacity"]

    coords_tensor = torch.tensor(coords, dtype=torch.float32)
    min_xy = coords_tensor.min(dim=0)[0]
    max_xy = coords_tensor.max(dim=0)[0]
    denom = (max_xy - min_xy).clamp(min=1.0)
    norm_coords = (coords_tensor - min_xy) / denom

    optimal_cost = float("nan")
    if load_solution:
        if sol_dir is not None:
            sol_path = os.path.join(sol_dir, f"{instance_name}.sol")
        else:
            sol_path = filepath.replace(".vrp", ".sol")
        if os.path.exists(sol_path):
            sol_data = vrplib.read_solution(sol_path)
            optimal_cost = sol_data.get("cost", float("nan"))

    return {
        "name": instance_name,
        "raw_coords": coords,
        "raw_demand": demand,
        "capacity": capacity,
        "norm_coords": norm_coords,
        "n_nodes": len(coords),
        "optimal_cost": optimal_cost,
    }


def save_results(df: pd.DataFrame, base_path: str, filename: str) -> str:
    """
    Save df to base_path/filename.csv with automatic versioning.

    If filename.csv already exists, writes filename_2.csv, filename_3.csv, etc.
    Returns the path actually written.
    """
    os.makedirs(base_path, exist_ok=True)
    out_path = os.path.join(base_path, f"{filename}.csv")
    if os.path.exists(out_path):
        counter = 2
        while os.path.exists(os.path.join(base_path, f"{filename}_{counter}.csv")):
            counter += 1
        out_path = os.path.join(base_path, f"{filename}_{counter}.csv")
    df.to_csv(out_path, index=False)
    return out_path
