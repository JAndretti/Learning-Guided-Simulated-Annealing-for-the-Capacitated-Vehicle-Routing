import argparse
import os
import sys
import time

import glob2
import numpy as np
import pandas as pd
import torch
import vrplib
from tqdm import tqdm

# --- Project Imports ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from func import (
    init_problem_parameters,
    load_model,
    set_seed,
)

from init import inf_test_model, initialize_models
from problem import CVRP
from utils import setup_logging

# --- Configuration & Setup ---
logger = setup_logging()
parser = argparse.ArgumentParser(description="Evaluate LGSA on CVRPLib Set X")

parser.add_argument(
    "--FOLDER", type=str, default="BEST", help="Folder containing the model"
)
parser.add_argument(
    "--DATA_PATH", type=str, default="bdd/X", help="Path to CVRPLib .vrp files"
)
parser.add_argument("--INIT", type=str, default="random", help="Initialization method")
parser.add_argument(
    "--OUTER_STEPS", type=int, default=10000, help="Number of steps for LGSA"
)
parser.add_argument("--seed", type=int, default=1234, help="Random seed")
parser.add_argument("--device", type=str, default="cpu")

args = parser.parse_args()


def calculate_cvrplib_cost(solution_indices, raw_coords):
    """
    Calculates the cost of a solution using CVRPLib conventions
    (Euclidean distance rounded to the nearest integer per edge).
    """
    distance = 0
    # Ensure solution starts and ends with depot (0) if not present
    # We assume 'solution_indices' is a list of node indices visiting the depot between routes

    # Simple check: if the solution is just a permutation of customers,
    # the cost calculation depends on how 'best_x' represents the tour.
    # Assuming 'best_x' is a full sequence including depots (0).

    for i in range(len(solution_indices) - 1):
        u = solution_indices[i]
        v = solution_indices[i + 1]

        c1 = raw_coords[u]
        c2 = raw_coords[v]

        # Euclidean distance
        exact_dist = np.sqrt(np.sum((c1 - c2) ** 2))

        # CVRPLib standard: Round to nearest integer
        distance += int(exact_dist + 0.5)

    return distance


def load_instance_data(filepath):
    """
    Loads a .vrp file and prepares normalized tensors for the model
    and raw data for final evaluation.
    """
    instance_name = os.path.basename(filepath).replace(".vrp", "")
    vrp_data = vrplib.read_instance(filepath)

    # Try loading solution if available
    sol_path = filepath.replace(".vrp", ".sol")
    if os.path.exists(sol_path):
        sol_data = vrplib.read_solution(sol_path)
        optimal_cost = sol_data.get("cost", float("nan"))
    else:
        optimal_cost = float("nan")

    # Raw Data
    coords = vrp_data["node_coord"]
    demand = vrp_data["demand"]
    capacity = vrp_data["capacity"]

    # --- Normalization for Model Input ---
    # 1. Normalize Coordinates (Min-Max scaling to [0, 1])
    # Note: We compute min/max from the specific instance
    coords_tensor = torch.tensor(coords, dtype=torch.float32)
    min_xy = torch.min(coords_tensor, dim=0)[0]
    max_xy = torch.max(coords_tensor, dim=0)[0]

    # Avoid division by zero if max == min
    denom = max_xy - min_xy
    denom[denom == 0] = 1.0

    normalized_coords = (coords_tensor - min_xy) / denom

    return {
        "name": instance_name,
        "raw_coords": coords,
        "raw_demand": demand,
        "capacity": capacity,
        "optimal_cost": optimal_cost,
        "norm_coords": normalized_coords,  # Shape: (N, 2)
    }


def solve_instance(model_name, instance_path, global_config):
    """
    Runs the model on a single CVRPLib instance.
    """
    # 1. Load Data
    data = load_instance_data(instance_path)

    # Prepare Batched Tensors (Batch Size = 1)
    # The model expects inputs of shape (Batch, N, ...)
    input_coords = data["norm_coords"].unsqueeze(0).to(args.device)  # (1, N, 2)

    N_nodes = input_coords.shape[1]

    # 2. Initialize Model (Actor)
    # We assume we can load the model parameters once, but for simplicity/safety
    # against changing dimensions (hidden layers etc), we init per model call.
    HP = init_problem_parameters(model_name, global_config)

    # !! CRITICAL ADAPTATION !!
    # We must construct a 'problem' object that holds this specific instance's data.
    problem = CVRP(
        dim=N_nodes - 1,
        n_problems=1,
        device=args.device,
        params=HP,
    )

    problem.manual_seed(args.seed)
    problem.set_heuristic(HP["HEURISTIC"])
    problem.set_feature_flags(HP["features"])
    input_dim = problem.get_input_dim()

    problem.generate_params(
        data["norm_coords"].unsqueeze(0),
        torch.tensor(data["raw_demand"]).unsqueeze(0),
        torch.tensor(data["capacity"]).unsqueeze(-1),
    )

    init_x = problem.generate_init_state(args.INIT, False)

    # 3. Load Network
    actor, _ = initialize_models(
        model_type=HP["MODEL"],
        critic_type="ff",
        embedding_dim=HP["EMBEDDING_DIM"],
        entry=input_dim,
        num_h_layers=HP["NUM_H_LAYERS"],
        update_method=HP["UPDATE_METHOD"],
        heuristic=HP["HEURISTIC"],
        seed=args.seed,
        device=args.device,
        attn_dim=HP.get("ATTN_DIM", 64),
        attn_num_heads=HP.get("ATTN_NUM_HEADS", 4),
        attn_num_layers=HP.get("ATTN_NUM_LAYERS", 1),
    )
    actor = load_model(actor, model_name, "actor").to(args.device)

    # 5. Run LGSA Inference
    HP["TEST_OUTER_STEPS"] = args.OUTER_STEPS

    start_time = time.time()
    results = inf_test_model(
        actor=actor,
        problem=problem,
        initial_solutions=init_x,
        config=HP,
        baseline=False,  # No baseline for evaluation mode
        greedy=False,
    )
    duration = time.time() - start_time

    # 6. Extract Solution and Calculate Real Cost
    best_solution_tensor = results["best_x"].squeeze(0).cpu().numpy()

    # Calculate cost using REAL coordinates and CVRPLib rounding
    lgsa_cost = calculate_cvrplib_cost(best_solution_tensor, data["raw_coords"])

    return {
        "Instance": data["name"],
        "Nodes": N_nodes,
        "Optimal_Cost": data["optimal_cost"],
        "LGSA_Cost": lgsa_cost,
        "Gap_Percent": 100 * (lgsa_cost - data["optimal_cost"]) / data["optimal_cost"]
        if not pd.isna(data["optimal_cost"])
        else float("nan"),
        "Time_sec": duration,
        "Steps": args.OUTER_STEPS,
    }


def main():
    set_seed(args.seed)

    # Setup Paths
    base_path = f"res/{args.FOLDER}/"
    os.makedirs(base_path, exist_ok=True)

    # Find Model (Assuming we pick the first one in the folder or a specific one)
    # If you want to evaluate multiple models, loop here.
    model_path_search = os.path.join("wandb", "LGSA", args.FOLDER, "models", "*")
    model_files = glob2.glob(model_path_search)

    if not model_files:
        print(f"No models found in {model_path_search}")
        return

    # Use the last saved model (or change logic to select specific)
    model_name = model_files[-1]
    print(f"Evaluating Model: {model_name}")

    # Config for HP loading
    cfg = {
        "PROBLEM_DIM": 100,  # Placeholder, will be overwritten per instance
        "DEVICE": args.device,
        "SEED": args.seed,
        "LOAD_PB": True,
        "INIT": args.INIT,
        "DATA": "cvrplib",
        "BASELINE": False,
    }

    # Find Instances
    instance_files = glob2.glob(os.path.join(args.DATA_PATH, "*.vrp"))
    instance_files.sort()  # Sort to keep order

    if not instance_files:
        print(f"No .vrp files found in {args.DATA_PATH}")
        return

    final_results = []

    print(f"Found {len(instance_files)} instances. Starting evaluation...")
    print("Warming up the model on a sample instance...")

    solve_instance(model_name, instance_files[len(instance_files) // 2], cfg)

    for f in tqdm(instance_files):
        try:
            res = solve_instance(model_name, f, cfg)
            final_results.append(res)

            # Print intermediate result
            gap_str = (
                f"{res['Gap_Percent']:.2f}%"
                if not pd.isna(res["Gap_Percent"])
                else "N/A"
            )
            tqdm.write(
                f"{res['Instance']} (N={res['Nodes']}): Opt={res['Optimal_Cost']} | LGSA={res['LGSA_Cost']} | Gap={gap_str}"
            )

        except Exception as e:
            print(f"Error processing {f}: {e}")
            import traceback

            traceback.print_exc()

    # Save to CSV
    df = pd.DataFrame(final_results)
    out_file = os.path.join(base_path, "cvrplib_X_results.csv")
    counter = 1
    while os.path.exists(out_file):
        name, ext = os.path.splitext(out_file)
        out_file = f"{name.rsplit('_', 1)[0]}_results_{counter}{ext}"
        counter += 1
    df.to_csv(out_file, index=False)

    print("\n--- Summary ---")
    print(f"Results saved to {out_file}")
    print(f"Average Gap: {df['Gap_Percent'].mean():.2f}%")


if __name__ == "__main__":
    main()
