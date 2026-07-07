import glob
import os
import random
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from rich import print

EXAMPLE_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.abspath(os.path.join(EXAMPLE_DIR, "..", "src")))
from model import CVRPActor
from problem import CVRP
from sa import sa_train
from utils import plot_vehicle_routes, prepare_plot


def set_seed(seed=0):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


if __name__ == "__main__":
    set_seed(1)

    cfg = {
        "PROBLEM_DIM": 100,
        "N_PROBLEMS": 1,
        "OUTER_STEPS": 10000,
        # "DEVICE": (
        #     "cuda"
        #     if torch.cuda.is_available()
        #     else "mps"
        #     if torch.backends.mps.is_available()
        #     else "cpu"
        # ),
        "DEVICE": "cpu",
        "SEED": 1,
        "LOAD_PB": True,
        "INIT": "random",
        "MULTI_INIT": False,
        "BASELINE": False,
        "GREEDY": False,
    }
    print("Using device:", cfg["DEVICE"])
    LOAD = {50: 40, 100: 50}
    cfg["MAX_LOAD"] = LOAD[cfg["PROBLEM_DIM"]]

    model_path = sorted(glob.glob(os.path.join(EXAMPLE_DIR, "models", "*")))[0]

    # Load the config saved alongside the checkpoint. It is serialized as an
    # `_HP` singleton (`!!python/object:setup.HP._HP`) with the actual settings
    # nested under a `config:` key. Strip the tag and pull out the flat dict.
    hp_file = os.path.join(model_path, "HP.yaml")
    with open(hp_file) as file:
        content = file.read().replace("!!python/object:setup.HP._HP", "")
        hp_data = yaml.safe_load(content)["config"]

    # generate data
    coords = torch.rand(cfg["N_PROBLEMS"], cfg["PROBLEM_DIM"] + 1, 2, device=cfg["DEVICE"])
    demands = torch.randint(
        1, 10, (cfg["N_PROBLEMS"], cfg["PROBLEM_DIM"] + 1), device=cfg["DEVICE"]
    )
    demands[:, 0] = 0  # Depot has no demand
    capacity = torch.full((cfg["N_PROBLEMS"], 1), cfg["MAX_LOAD"], device=cfg["DEVICE"])

    hp_data.update(cfg)
    cfg = hp_data

    problem = CVRP(
        dim=cfg["PROBLEM_DIM"],
        n_problems=cfg["N_PROBLEMS"],
        device=cfg["DEVICE"],
        params=cfg,
    )
    problem.manual_seed(0)
    problem.set_heuristic(hp_data["HEURISTIC"])
    problem.set_feature_flags(hp_data["features"])
    input_dim = problem.get_input_dim()

    problem.generate_params(coords, demands, capacity)

    # get model (mirror the construction in src/init.py / src/main.py)
    actor = CVRPActor(
        embed_dim=hp_data["EMBEDDING_DIM"],
        c=input_dim,
        num_hidden_layers=hp_data["NUM_H_LAYERS"],
        device=cfg["DEVICE"],
        method=hp_data["UPDATE_METHOD"],
        cond_rank=hp_data.get("COND_RANK", False),
        cond_detour=hp_data.get("COND_DETOUR", False),
        global_context=hp_data.get("GLOBAL_CONTEXT", False),
        logit_clip=hp_data.get("LOGIT_CLIP", 0.0),
        learnable_temp=hp_data.get("LEARNABLE_TEMP", False),
    )
    checkpoint = sorted(glob.glob(os.path.join(model_path, "*.pt")))[0]
    actor.load_state_dict(
        torch.load(
            checkpoint,
            map_location=torch.device("cpu"),
            weights_only=True,
        )
    )
    actor.to(cfg["DEVICE"])

    # Generate initial solution for problems
    init_x = problem.generate_init_state(cfg["INIT"], False)
    init_x_cost = torch.mean(problem.cost(init_x)).item()
    print(f"Initial solution cost: {init_x_cost:.2f}")
    print(
        f"Min cost: {torch.min(problem.cost(init_x)).item():.2f}, Max cost: {torch.max(problem.cost(init_x)).item():.2f}"
    )
    init_nearest_neighbor = problem.generate_init_state("nearest_neighbor", False)
    init_nearest_cost = torch.mean(problem.cost(init_nearest_neighbor)).item()
    print(f"Nearest Neighbor solution cost: {init_nearest_cost:.2f}")
    print(
        f"Min cost: {torch.min(problem.cost(init_nearest_neighbor)).item():.2f}, Max cost: {torch.max(problem.cost(init_nearest_neighbor)).item():.2f}"
    )
    init_CW = problem.generate_init_state("Clark_and_Wright", False)
    init_CW_cost = torch.mean(problem.cost(init_CW)).item()
    print(f"Clark and Wright solution cost: {init_CW_cost:.2f}")
    print(
        f"Min cost: {torch.min(problem.cost(init_CW)).item():.2f}, Max cost: {torch.max(problem.cost(init_CW)).item():.2f}"
    )

    # Save initial plots
    plots_dir = os.path.join(EXAMPLE_DIR, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    data_init, sol_init = prepare_plot(problem, init_x)
    data_init_n, sol_init_n = prepare_plot(problem, init_nearest_neighbor)
    data_init_CW, sol_init_CW = prepare_plot(problem, init_CW)

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
            title=f"Initial Solution for Instance {i + 1} / ",
        )
        plt.savefig(os.path.join(plots_dir, f"instance_{i + 1}_init.png"))
        plt.close()
        instance_nearest_sol = sol_init_n[i]
        fig, ax1 = plt.subplots(figsize=(10, 10))
        plot_vehicle_routes(
            instance_data,
            instance_nearest_sol,
            ax1=ax1,
            capacity=cfg["MAX_LOAD"],
            title=f"Solution with Nearest Neighbor for Instance {i + 1} / ",
        )
        plt.savefig(os.path.join(plots_dir, f"instance_{i + 1}_init_nearest.png"))
        plt.close()
        instance_CW_sol = sol_init_CW[i]
        fig, ax1 = plt.subplots(figsize=(10, 10))
        plot_vehicle_routes(
            instance_data,
            instance_CW_sol,
            ax1=ax1,
            capacity=cfg["MAX_LOAD"],
            title=f"Solution with Clark and Wright for Instance {i + 1} / ",
        )
        plt.savefig(os.path.join(plots_dir, f"instance_{i + 1}_init_CW.png"))
        plt.close()

    with torch.no_grad():
        cfg["TEST_OUTER_STEPS"] = cfg["OUTER_STEPS"]
        init_x = problem.generate_init_state(cfg["INIT"], False)
        # sa_train returns (results_td, results_extra); we only need the TensorDict.
        res, _ = sa_train(
            actor,
            problem,
            init_x,
            cfg,
            baseline=cfg["BASELINE"],
            greedy=cfg["GREEDY"],
            desc_tqdm="LGSA Model Evaluation",
        )

    best_costs = res["min_cost"]
    print(f"Best solution costs: {torch.mean(best_costs).item():.2f}")
    print(
        f"Min cost: {torch.min(best_costs).item():.2f}, Max cost: {torch.max(best_costs).item():.2f}"
    )
    best_solutions = res["best_x"]

    data, sol = prepare_plot(problem, best_solutions)

    for i in range(cfg["N_PROBLEMS"]):
        print(f"Creating plot {i + 1} / {cfg['N_PROBLEMS']}...")
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
            title=f"LGSA Solution for Instance {i + 1} / ",
        )
        plt.savefig(os.path.join(plots_dir, f"instance_{i + 1}.png"))
        plt.close()
    print(f"All plots saved in {plots_dir}/")
