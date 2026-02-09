import os
import random
import sys

import glob2
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from rich import print

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
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
        "PROBLEM_DIM": 50,
        "N_PROBLEMS": 10,
        "OUTER_STEPS": 10000,
        "DEVICE": (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        ),
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

    model_path = glob2.glob("example/models/*")[0]

    # get HP model
    hp_file = os.path.join(model_path, "HP.yaml")
    with open(hp_file, "r") as file:
        content = file.read()
        content_clean = content.replace("!!python/object:HP._HP", "")
        hp_data = yaml.unsafe_load(content_clean)

    # generate data
    coords = torch.rand(
        cfg["N_PROBLEMS"], cfg["PROBLEM_DIM"] + 1, 2, device=cfg["DEVICE"]
    )
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

    # get model
    actor = CVRPActor(
        embed_dim=hp_data["EMBEDDING_DIM"],
        c=input_dim,
        num_hidden_layers=hp_data["NUM_H_LAYERS"],
        device=cfg["DEVICE"],
        mixed_heuristic=False,
        method=hp_data["UPDATE_METHOD"],
    )
    actor.load_state_dict(
        torch.load(
            os.path.join(
                model_path,
                "LGSA_CRITIC_actor_epoch_180_loss_18.121649.pt",
            ),
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
    if not os.path.exists("example/plots"):
        os.makedirs("example/plots")

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
            title="Initial Solution for Instance {} / ".format(i + 1),
        )
        plt.savefig(f"example/plots/instance_{i + 1}_init.png")
        plt.close()
        instance_nearest_sol = sol_init_n[i]
        fig, ax1 = plt.subplots(figsize=(10, 10))
        plot_vehicle_routes(
            instance_data,
            instance_nearest_sol,
            ax1=ax1,
            capacity=cfg["MAX_LOAD"],
            title="Solution with Nearest Neighbor for Instance {} / ".format(i + 1),
        )
        plt.savefig(f"example/plots/instance_{i + 1}_init_nearest.png")
        plt.close()
        instance_CW_sol = sol_init_CW[i]
        fig, ax1 = plt.subplots(figsize=(10, 10))
        plot_vehicle_routes(
            instance_data,
            instance_CW_sol,
            ax1=ax1,
            capacity=cfg["MAX_LOAD"],
            title="Solution with Clark and Wright for Instance {} / ".format(i + 1),
        )
        plt.savefig(f"example/plots/instance_{i + 1}_init_CW.png")
        plt.close()

    with torch.no_grad():
        cfg["TEST_OUTER_STEPS"] = cfg["OUTER_STEPS"]
        init_x = problem.generate_init_state(cfg["INIT"], False)
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
            title="LGSA Solution for Instance {} / ".format(i + 1),
        )
        plt.savefig(f"example/plots/instance_{i + 1}.png")
        plt.close()
    print("All plots saved in example/plots/")
