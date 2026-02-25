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

from init import initialize_models, test_model
from problem import CVRP
from utils import setup_logging

# --- Configuration & Setup ---
logger = setup_logging()
parser = argparse.ArgumentParser(description="Evaluate LGSA on CVRPLib Set XL")

parser.add_argument(
    "--FOLDER", type=str, default="BEST", help="Folder containing the model"
)
# Changed default path to XL
parser.add_argument(
    "--DATA_PATH", type=str, default="bdd/XL", help="Path to CVRPLib XL .vrp files"
)
parser.add_argument("--INIT", type=str, default="random", help="Initialization method")
parser.add_argument(
    "--OUTER_STEPS", type=int, default=1000, help="Number of steps for LGSA"
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
    Loads a .vrp file and prepares normalized tensors for the model.
    Optimized to skip .sol file checks since they are unavailable.
    """
    instance_name = os.path.basename(filepath).replace(".vrp", "")
    vrp_data = vrplib.read_instance(filepath)

    # Raw Data
    coords = vrp_data["node_coord"]
    demand = vrp_data["demand"]
    capacity = vrp_data["capacity"]

    # --- Normalization for Model Input ---
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
        "norm_coords": normalized_coords,  # Shape: (N, 2)
    }


def solve_instance(model_name, instance_path, global_config):
    """
    Runs the model on a single CVRPLib XL instance.
    """
    # 1. Load Data
    data = load_instance_data(instance_path)

    # Prepare Batched Tensors (Batch Size = 1)
    input_coords = data["norm_coords"].unsqueeze(0).to(args.device)  # (1, N, 2)

    N_nodes = input_coords.shape[1]

    # 2. Initialize Model (Actor)
    HP = init_problem_parameters(model_name, global_config)

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
    )
    actor = load_model(actor, model_name, "actor").to(args.device)

    # 4. Run LGSA Inference
    HP["TEST_OUTER_STEPS"] = args.OUTER_STEPS

    start_time = time.time()
    results = test_model(
        actor=actor,
        problem=problem,
        initial_solutions=init_x,
        config=HP,
        baseline=False,
        greedy=False,
    )
    duration = time.time() - start_time

    # 5. Extract Solution and Calculate Real Cost
    best_solution_tensor = results["best_x"].squeeze(0).cpu().numpy()

    # Calculate cost using REAL coordinates and CVRPLib rounding
    lgsa_cost = calculate_cvrplib_cost(best_solution_tensor, data["raw_coords"])

    # Removed Optimal_Cost and Gap_Percent since they don't apply here
    return {
        "Instance": data["name"],
        "Nodes": N_nodes,
        "LGSA_Cost": lgsa_cost,
        "Time_sec": duration,
        "Steps": args.OUTER_STEPS,
    }


def main():
    set_seed(args.seed)

    # Setup Paths
    base_path = f"res/{args.FOLDER}/"
    os.makedirs(base_path, exist_ok=True)

    # Find Model
    model_path_search = os.path.join("wandb", "LGSA", args.FOLDER, "models", "*")
    model_files = glob2.glob(model_path_search)

    if not model_files:
        print(f"No models found in {model_path_search}")
        return

    model_name = model_files[-1]
    print(f"Evaluating Model: {model_name}")

    # Config for HP loading
    cfg = {
        "PROBLEM_DIM": 100,
        "DEVICE": args.device,
        "SEED": args.seed,
        "LOAD_PB": True,
        "INIT": args.INIT,
        "DATA": "cvrplib",
        "BASELINE": False,
    }

    # Find Instances
    instance_files = glob2.glob(os.path.join(args.DATA_PATH, "*.vrp"))
    instance_files.sort()

    if not instance_files:
        print(f"No .vrp files found in {args.DATA_PATH}")
        return

    final_results = []

    print(f"Found {len(instance_files)} instances. Starting evaluation...")

    for f in tqdm(instance_files):
        try:
            res = solve_instance(model_name, f, cfg)
            final_results.append(res)

            # Updated print statement for missing gap
            tqdm.write(
                f"{res['Instance']} (N={res['Nodes']}): LGSA_Cost={res['LGSA_Cost']} | Time={res['Time_sec']:.2f}s"
            )

        except Exception as e:
            print(f"Error processing {f}: {e}")
            import traceback

            traceback.print_exc()

    # Save to CSV
    df = pd.DataFrame(final_results)

    # Updated output file name
    out_file = os.path.join(base_path, "cvrplib_XL_results.csv")
    df.to_csv(out_file, index=False)

    print("\n--- Summary ---")
    print(f"Results saved to {out_file}")

    # Swapped Average Gap out for Average Time & Cost as fallback metrics
    if not df.empty:
        print(f"Average LGSA Cost: {df['LGSA_Cost'].mean():.2f}")
        print(f"Average Time per Instance: {df['Time_sec'].mean():.2f}s")


if __name__ == "__main__":
    main()
