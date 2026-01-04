import os
import random
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from model import SAModel
from problem import CVRP
from utils import setup_logging

logger = setup_logging()


# Load model hyperparameters
def get_HP_for_model(model_name):
    """Load hyperparameters for a given model."""
    hp_file = os.path.join(model_name, "HP.yaml")
    with open(hp_file, "r") as file:
        content = file.read()
        content_clean = content.replace("!!python/object:HP._HP", "")
        hp_data = yaml.unsafe_load(content_clean)
        return hp_data.config


def init_problem_parameters(model_path: str, cfg: dict):
    logger.info("Initializing problem parameters...")
    HP = get_HP_for_model(model_path)
    HP.update(cfg)
    logger.info("Problem parameters initialized.")
    return HP


def set_seed(seed=0):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# Load model weights
def extract_loss(filename):
    """Extract loss value from filename."""
    try:
        return float((filename.split("_")[-1])[:-3])
    except ValueError:
        return float("inf")


# Load model with smallest loss
def load_model(model: torch.nn.Module, folder: str, special_key: str = "") -> SAModel:
    """Load model with smallest loss."""
    files = [f for f in os.listdir(folder) if f.endswith(".pt")]
    if len(special_key) > 0:
        files = [f for f in files if special_key in f]
    if files:
        best_file = min(files, key=extract_loss)
        model.load_state_dict(
            torch.load(os.path.join(folder, best_file), weights_only=True)
        )
    return model


def init_pb(
    cfg: dict,
    coords: torch.Tensor,
    demands: torch.Tensor,
    capacities: torch.Tensor,
):
    """Initialize the CVRP problem and generate initial solutions."""
    logger.info("Initializing CVRP problem...")

    problem = CVRP(
        cfg["PROBLEM_DIM"],
        cfg["N_PROBLEMS"],
        capacities,
        device=cfg["DEVICE"],
        params=cfg,
    )
    problem.manual_seed(cfg["SEED"])
    problem.generate_params(
        mode="test",
        pb=True,
        coords=coords,
        demands=demands,
    )
    return problem


def plot_cvrp_solution(
    coords: torch.Tensor,
    solution: torch.Tensor,
    title: str = "CVRP Solution",
    show_routes: bool = True,
):
    """
    Plot a CVRP solution.

    Args:
        coords: Tensor of shape [batch_size, num_nodes, 2] containing node coordinates
        solution: Tensor of shape [batch_size, route_length, 1] containing the solution
        title: Title for the plot

    Returns:
        fig: Matplotlib figure object that can be displayed later with plt.show()
    """
    # Extract the first problem in the batch for plotting
    coords = coords[0].cpu().numpy()  # [num_nodes, 2]

    # Create figure
    fig = plt.figure(figsize=(10, 8))
    plt.scatter(coords[:, 0], coords[:, 1], s=50, c="blue", label="Nodes")

    # Highlight depot
    plt.scatter(coords[0, 0], coords[0, 1], s=100, c="red", marker="*", label="Depot")

    if show_routes:
        # Plot routes
        route_idx = 0
        colors = [
            "green",
            "purple",
            "orange",
            "brown",
            "pink",
            "gray",
            "olive",
            "cyan",
            "magenta",
            "crimson",
            "teal",
            "navy",
            "gold",
            "indigo",
            "maroon",
            "lime",
            "coral",
            "salmon",
            "turquoise",
            "violet",
            "orchid",
            "slateblue",
            "firebrick",
            "tomato",
        ]
        solution = solution[0, :, 0].cpu().numpy()  # [route_length]
        # Filter out zeros at the end of solution
        last_nonzero = np.max(np.where(solution > 0)[0]) if np.any(solution > 0) else 0
        solution = solution[: last_nonzero + 1]

        current_route = []
        current_color_idx = 0

        for node_idx in solution:
            current_route.append(int(node_idx))

            # If we return to depot, plot the current route and reset
            if node_idx == 0 and len(current_route) > 1:
                route_coords = coords[current_route]
                color = colors[current_color_idx % len(colors)]
                plt.plot(route_coords[:, 0], route_coords[:, 1], c=color, linewidth=2)

                # Reset for next route
                current_route = [0]  # Start next route from depot
                current_color_idx += 1
                route_idx += 1

        # Plot the last route if it doesn't end at the depot
        if len(current_route) > 1:  # Contains more than just depot
            # Add depot at the end if not already there
            if current_route[-1] != 0:
                current_route.append(0)

            route_coords = coords[current_route]
            color = colors[current_color_idx % len(colors)]
            plt.plot(route_coords[:, 0], route_coords[:, 1], c=color, linewidth=2)

    # Add labels to nodes
    for i in range(len(coords)):
        plt.annotate(
            str(i),
            (coords[i, 0], coords[i, 1]),
            xytext=(5, 5),
            textcoords="offset points",
        )

    plt.title(title)
    plt.legend()
    plt.grid(True)
    return fig
