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
parser = argparse.ArgumentParser(
    description="Evaluate LGSA on Queiroga (XML) Set (Batch Mode)"
)

parser.add_argument(
    "--FOLDER", type=str, default="BEST", help="Folder containing the model"
)
parser.add_argument(
    "--DATA_PATH", type=str, default="bdd/XML", help="Path to .vrp files"
)
parser.add_argument(
    "--SOL_PATH", type=str, default="bdd/solutions", help="Path to .sol files"
)
parser.add_argument("--INIT", type=str, default="random", help="Initialization method")
parser.add_argument(
    "--OUTER_STEPS", type=int, default=1000, help="Number of steps for LGSA"
)
parser.add_argument("--seed", type=int, default=1234, help="Random seed")
parser.add_argument(
    "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
)

args = parser.parse_args()


def calculate_exact_cost(solution_indices, raw_coords):
    """
    Calculates exact Euclidean distance (float) for the Queiroga Set.
    (Unlike Set X, these usually do not require rounding to nearest int).
    """
    distance = 0
    for i in range(len(solution_indices) - 1):
        u = solution_indices[i]
        v = solution_indices[i + 1]
        c1 = raw_coords[u]
        c2 = raw_coords[v]
        distance += np.sqrt(np.sum((c1 - c2) ** 2))
    return distance


def solve_batch(model_name, instance_files, global_config):
    """
    Loads ALL instances, batches them, runs the model once,
    and evaluates results.
    """
    print(f"Loading {len(instance_files)} instances for batch processing...")

    batch_data = []

    # 1. Load and Normalize Data
    # We read the first file to establish expected dimensions
    first_instance = vrplib.read_instance(instance_files[0])
    expected_dim = len(first_instance["node_coord"])

    print(f"Expecting problem dimension: {expected_dim} nodes (including depot)")

    for f in tqdm(instance_files, desc="Loading VRPs"):
        # Load Instance
        data = vrplib.read_instance(f)
        instance_name = os.path.basename(f).replace(".vrp", "")

        # Verify Dimension
        if len(data["node_coord"]) != expected_dim:
            print(
                f"Skipping {instance_name}: Dimension {len(data['node_coord'])} != {expected_dim}"
            )
            continue

        # Load Solution (if exists)
        sol_file = os.path.join(args.SOL_PATH, f"{instance_name}.sol")
        if os.path.exists(sol_file):
            sol_data = vrplib.read_solution(sol_file)
            opt_cost = sol_data.get("cost", float("nan"))
        else:
            opt_cost = float("nan")

        # Normalize Coords
        coords_tensor = torch.tensor(data["node_coord"], dtype=torch.float32)
        min_xy = torch.min(coords_tensor, dim=0)[0]
        max_xy = torch.max(coords_tensor, dim=0)[0]
        denom = max_xy - min_xy
        denom[denom == 0] = 1.0
        norm_coords = (coords_tensor - min_xy) / denom

        batch_data.append(
            {
                "name": instance_name,
                "opt_cost": opt_cost,
                "norm_coords": norm_coords,
                "raw_coords": data["node_coord"],
                "raw_demand": torch.tensor(data["demand"]),
                "capacity": data["capacity"],
            }
        )

    # Stack into Batch Tensors
    # Shape: (Batch_Size, N_Nodes, 2)
    batch_coords = torch.stack([b["norm_coords"] for b in batch_data]).to(args.device)
    batch_demands = torch.stack([b["raw_demand"] for b in batch_data]).to(args.device)

    # Capacity is implicitly 1.0 in normalized space, but we pass raw capacity for the Problem class
    batch_raw_capacity = (
        torch.tensor([b["capacity"] for b in batch_data], dtype=torch.float32)
        .unsqueeze(-1)
        .to(args.device)
    )

    batch_size = len(batch_data)
    n_nodes = batch_coords.shape[1]

    print(f"Created batch of size {batch_size} with {n_nodes} nodes.")

    # 2. Initialize Model & Problem
    HP = init_problem_parameters(model_name, global_config)

    # Initialize Problem with n_problems = batch_size
    problem = CVRP(
        dim=n_nodes - 1,  # Number of customers
        n_problems=batch_size,
        device=args.device,
        params=HP,
    )

    problem.manual_seed(args.seed)
    problem.set_heuristic(HP["HEURISTIC"])
    problem.set_feature_flags(HP["features"])
    input_dim = problem.get_input_dim()

    # Inject Batch Data
    problem.generate_params(batch_coords, batch_demands, batch_raw_capacity)

    # Generate Initial Solution (Batch)
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

    # 4. Run LGSA Inference (One Pass)
    HP["TEST_OUTER_STEPS"] = args.OUTER_STEPS
    print("Running LGSA Inference on batch...")

    start_time = time.time()
    results = test_model(
        actor=actor,
        problem=problem,
        initial_solutions=init_x,
        config=HP,
        baseline=False,
        greedy=False,
    )
    total_duration = time.time() - start_time
    print(
        f"Inference finished in {total_duration:.2f}s ({(total_duration / batch_size):.4f}s per instance)"
    )

    # 5. Process Results
    final_results = []

    # best_x shape: (Batch, Sequence_Length)
    best_solutions = results["best_x"].cpu().numpy()

    for i in range(batch_size):
        sol_indices = best_solutions[i]
        meta = batch_data[i]

        # Calculate Real Cost
        lgsa_cost = calculate_exact_cost(sol_indices, meta["raw_coords"])

        opt_cost = meta["opt_cost"]
        gap = (
            100 * (lgsa_cost - opt_cost) / opt_cost
            if not pd.isna(opt_cost)
            else float("nan")
        )

        final_results.append(
            {
                "Instance": meta["name"],
                "Nodes": n_nodes,
                "Optimal_Cost": opt_cost,
                "LGSA_Cost": lgsa_cost,
                "Gap_Percent": gap,
                "Time_sec": total_duration / batch_size,  # Amortized time
            }
        )

    return final_results


def main():
    set_seed(args.seed)

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

    # Config
    cfg = {
        "PROBLEM_DIM": 100,
        "DEVICE": args.device,
        "SEED": args.seed,
        "LOAD_PB": True,
        "INIT": args.INIT,
        "DATA": "queiroga_xml",
        "BASELINE": False,
    }

    # Find VRP Files
    instance_files = glob2.glob(os.path.join(args.DATA_PATH, "*.vrp"))
    instance_files.sort()

    if not instance_files:
        print(f"No .vrp files found in {args.DATA_PATH}")
        return

    # Run Batch Solve
    try:
        final_results = solve_batch(model_name, instance_files, cfg)

        # Save Results
        df = pd.DataFrame(final_results)
        out_file = os.path.join(base_path, "queiroga_xml_results.csv")
        df.to_csv(out_file, index=False)

        print("\n--- Summary ---")
        # Print first 10 for sanity check
        print(
            df[["Instance", "Optimal_Cost", "LGSA_Cost", "Gap_Percent"]]
            .head(10)
            .to_string()
        )
        print(f"\nResults saved to {out_file}")
        print(f"Average Gap: {df['Gap_Percent'].mean():.2f}%")

    except Exception as e:
        print(f"Batch processing failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
