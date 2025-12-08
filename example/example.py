import os
import sys
import torch
import random
import numpy as np
import yaml
import matplotlib.pyplot as plt


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from sa import sa_train
from model import CVRPActor
from problem import CVRP
from utils import plot_vehicle_routes, prepare_plot


def set_seed(seed=0):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


if __name__ == "__main__":
    set_seed(1)

    cfg = {
        "PROBLEM_DIM": 50,
        "N_PROBLEMS": 10,
        "OUTER_STEPS": 10000,
        "DEVICE": (
            "cuda"
            if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu"
        ),
        "SEED": 1,
        "LOAD_PB": True,
        "INIT": "random",
        "MULTI_INIT": False,
        "BASELINE": False,
        "GREEDY": False,
    }
    LOAD = {50: 40, 100: 50}
    cfg["MAX_LOAD"] = LOAD[cfg["PROBLEM_DIM"]]

    model_path = "example/model/20251121_114610_3n73ugy5/"

    # get HP model
    hp_file = os.path.join(model_path, "HP.yaml")
    with open(hp_file, "r") as file:
        content = file.read()
        content_clean = content.replace("!!python/object:HP._HP", "")
        hp_data = yaml.unsafe_load(content_clean)

    # get model
    actor = CVRPActor(
        embed_dim=hp_data["EMBEDDING_DIM"],
        c=hp_data["ENTRY"],
        num_hidden_layers=hp_data["NUM_H_LAYERS"],
        device=cfg["DEVICE"],
        mixed_heuristic=False,
        method=hp_data["UPDATE_METHOD"],
    )
    actor.load_state_dict(
        torch.load(
            os.path.join(
                model_path,
                "192_loss_19.997517.pt",
            ),
            weights_only=True,
        )
    )

    hp_data.update(cfg)
    cfg = hp_data

    problem = CVRP(
        dim=cfg["PROBLEM_DIM"],
        n_problems=cfg["N_PROBLEMS"],
        capacities=cfg["MAX_LOAD"],
        device=cfg["DEVICE"],
        params=cfg,
    )
    problem.manual_seed(0)
    problem.set_heuristic(hp_data["HEURISTIC"])
    problem.generate_params(mode="test")

    # Generate initial solution for problems
    init_x = problem.generate_init_state(cfg["INIT"], False)

    # Save initial plots
    if not os.path.exists("example/plots"):
        os.makedirs("example/plots")

    data_init, sol_init = prepare_plot(problem, init_x)
    for i in range(cfg["N_PROBLEMS"]):
        instance_depot = data_init["depot"][i]
        instance_loc = data_init["loc"][i]
        instance_demand = data_init["demand"][i]
        instance_data = {
            "depot": instance_depot,
            "loc": instance_loc,
            "demand": instance_demand,
        }
        instance_sol = sol_init[i]
        fig, ax1 = plt.subplots(figsize=(10, 10))
        plot_vehicle_routes(
            instance_data,
            instance_sol,
            ax1=ax1,
            capacity=cfg["MAX_LOAD"],
            title="Initial Solution for CVRP Instance {} / ".format(i + 1),
        )
        plt.savefig(f"example/plots/instance_{i + 1}_init.png")
        plt.close()

    with torch.no_grad():
        res = sa_train(
            actor,
            problem,
            init_x,
            cfg,
            baseline=cfg["BASELINE"],
            greedy=cfg["GREEDY"],
            desc_tqdm="LGSA Model Evaluation",
        )

    best_costs = res["min_cost"]
    best_solutions = res["best_x"]

    data, sol = prepare_plot(problem, best_solutions)

    for i in range(cfg["N_PROBLEMS"]):
        # Extract data for the i-th problem instance
        instance_depot = data["depot"][i]
        instance_loc = data["loc"][i]
        instance_demand = data["demand"][i]
        instance_data = {
            "depot": instance_depot,
            "loc": instance_loc,
            "demand": instance_demand,
        }
        instance_sol = sol[i]
        # You can now process or plot each instance individually here
        fig, ax1 = plt.subplots(figsize=(10, 10))
        plot_vehicle_routes(
            instance_data,
            instance_sol,
            ax1=ax1,
            capacity=cfg["MAX_LOAD"],
            title="LGSA Solution for CVRP Instance {} / ".format(i + 1),
        )
        plt.savefig(f"example/plots/instance_{i + 1}.png")
        plt.close()
