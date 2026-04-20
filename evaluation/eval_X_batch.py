# evaluation/eval_X_batch.py

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

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from func import init_problem_parameters, load_model, set_seed
from init import initialize_models, test_model
from problem import CVRP
from utils import setup_logging

logger = setup_logging()

parser = argparse.ArgumentParser(description="Batched evaluation of LGSA on CVRPLib Set X")
parser.add_argument("--FOLDER", type=str, default="BEST")
parser.add_argument("--DATA_PATH", type=str, default="bdd/X")
parser.add_argument("--INIT", type=str, default="random")
parser.add_argument("--OUTER_STEPS", type=int, default=10000)
parser.add_argument("--seed", type=int, default=1234)
parser.add_argument("--device", type=str, default="cpu")
parser.add_argument("--mode", type=str, default="bucket", choices=["bucket", "global"])
parser.add_argument("--n_buckets", type=int, default=5)
parser.add_argument("--batch_size", type=str, default="all")
args = parser.parse_args()


# ============================================================================
# DATA LOADING
# ============================================================================


def calculate_cvrplib_cost(solution_indices, raw_coords):
    """Euclidean distance rounded to nearest integer per edge (CVRPLib convention)."""
    distance = 0
    for i in range(len(solution_indices) - 1):
        u, v = solution_indices[i], solution_indices[i + 1]
        c1, c2 = raw_coords[u], raw_coords[v]
        distance += int(np.sqrt(np.sum((c1 - c2) ** 2)) + 0.5)
    return distance


def load_instance_data(filepath):
    instance_name = os.path.basename(filepath).replace(".vrp", "")
    vrp_data = vrplib.read_instance(filepath)

    sol_path = filepath.replace(".vrp", ".sol")
    optimal_cost = float("nan")
    if os.path.exists(sol_path):
        sol_data = vrplib.read_solution(sol_path)
        optimal_cost = sol_data.get("cost", float("nan"))

    coords = vrp_data["node_coord"]
    demand = vrp_data["demand"]
    capacity = vrp_data["capacity"]

    coords_tensor = torch.tensor(coords, dtype=torch.float32)
    min_xy = coords_tensor.min(dim=0)[0]
    max_xy = coords_tensor.max(dim=0)[0]
    denom = (max_xy - min_xy).clamp(min=1.0)
    normalized_coords = (coords_tensor - min_xy) / denom

    return {
        "name": instance_name,
        "raw_coords": coords,
        "raw_demand": demand,
        "capacity": capacity,
        "optimal_cost": optimal_cost,
        "norm_coords": normalized_coords,
        "n_nodes": len(coords),  # includes depot
    }


# ============================================================================
# BUCKETING & PADDING
# ============================================================================


def create_buckets(instances, mode, n_buckets):
    """Sort by N and split into percentile groups, or return one global bucket."""
    sorted_instances = sorted(instances, key=lambda x: x["n_nodes"])
    if mode == "global":
        return [sorted_instances]
    buckets = np.array_split(sorted_instances, n_buckets)
    return [list(b) for b in buckets if len(b) > 0]


def pad_instances_to_N(instances, max_N, device):
    """
    Pad each instance's coords and demands to max_N using ghost depot nodes.
    Ghost nodes have demand=0 and coords=depot, so the CVRP codebase treats
    them as extra depots (never selected as move sources, zero route load).

    Returns:
        coords:     [B, max_N, 2]
        demands:    [B, max_N]
        capacities: [B, 1]
    """
    coords_list, demands_list, caps_list = [], [], []

    for inst in instances:
        nc = inst["norm_coords"]       # [N_i, 2]
        N_i = nc.shape[0]
        pad = max_N - N_i

        if pad > 0:
            depot_rep = nc[0:1].expand(pad, -1)      # [pad, 2] — depot coords
            nc = torch.cat([nc, depot_rep], dim=0)   # [max_N, 2]

        dem = torch.tensor(inst["raw_demand"])        # [N_i], dtype inferred from data
        if pad > 0:
            dem = torch.cat([dem, torch.zeros(pad, dtype=dem.dtype)], dim=0)  # [max_N]

        coords_list.append(nc)
        demands_list.append(dem)
        caps_list.append(inst["capacity"])

    coords = torch.stack(coords_list).to(device)                                         # [B, max_N, 2]
    demands = torch.stack(demands_list).to(device)                                       # [B, max_N]
    capacities = torch.tensor(caps_list, dtype=torch.float32).unsqueeze(-1).to(device)  # [B, 1]
    return coords, demands, capacities


# ============================================================================
# COST EXTRACTION & SOLVING
# ============================================================================


def extract_instance_cost(solution, actual_N, raw_coords):
    """
    Remove ghost node indices (>= actual_N) from the solution and compute
    the CVRPLib integer cost using raw (unnormalized) coordinates.
    """
    sol_clean = [int(x) for x in solution if int(x) < actual_N]
    return calculate_cvrplib_cost(sol_clean, raw_coords)


def solve_bucket(bucket, bucket_id, max_N, actor, HP, args):
    """
    Run SA on all instances in a bucket in mini-batches.
    Returns a list of result dicts (one per instance).
    """
    results = []
    bs = len(bucket) if args.batch_size == "all" else int(args.batch_size)

    for start in range(0, len(bucket), bs):
        mini_batch = bucket[start : start + bs]
        B = len(mini_batch)

        coords, demands, capacities = pad_instances_to_N(mini_batch, max_N, args.device)

        problem = CVRP(dim=max_N - 1, n_problems=B, device=args.device, params=HP)
        problem.manual_seed(args.seed)
        problem.set_heuristic(HP["HEURISTIC"])
        problem.set_feature_flags(HP["features"])

        problem.generate_params(coords, demands, capacities)
        init_x = problem.generate_init_state(args.INIT, False)

        t0 = time.time()
        result = test_model(
            actor=actor,
            problem=problem,
            initial_solutions=init_x,
            config=HP,
            baseline=False,
            greedy=False,
        )
        elapsed = time.time() - t0

        best_x = result["best_x"].cpu().numpy()  # [B, seq_len, 1]

        for i, inst in enumerate(mini_batch):
            sol = best_x[i, :, 0].tolist()
            actual_N = inst["n_nodes"]
            cost = extract_instance_cost(sol, actual_N, inst["raw_coords"])
            opt = inst["optimal_cost"]
            gap = 100 * (cost - opt) / opt if not np.isnan(opt) else float("nan")

            results.append({
                "Instance": inst["name"],
                "Nodes": actual_N,
                "Optimal_Cost": opt,
                "LGSA_Cost": cost,
                "Gap_Percent": gap,
                "Time_sec": elapsed / B,
                "Steps": args.OUTER_STEPS,
                "Bucket_ID": bucket_id,
                "Batch_size_actual": B,
            })

    return results


# ============================================================================
# MAIN
# ============================================================================


def main():
    set_seed(args.seed)

    base_path = f"res/{args.FOLDER}/"
    os.makedirs(base_path, exist_ok=True)

    # Locate model
    model_path_search = os.path.join("wandb", "LGSA", args.FOLDER, "models", "*")
    model_files = glob2.glob(model_path_search)
    if not model_files:
        print(f"No models found in {model_path_search}")
        return
    model_name = model_files[-1]
    print(f"Evaluating Model: {model_name}")

    cfg = {
        "PROBLEM_DIM": 100,
        "DEVICE": args.device,
        "SEED": args.seed,
        "LOAD_PB": True,
        "INIT": args.INIT,
        "DATA": "cvrplib",
        "BASELINE": False,
    }
    HP = init_problem_parameters(model_name, cfg)
    HP["TEST_OUTER_STEPS"] = args.OUTER_STEPS  # set once; passed to all solve_bucket calls

    # Load actor once — input_dim depends only on feature flags, not on N
    _probe = CVRP(dim=100, n_problems=1, device=args.device, params=HP)
    _probe.set_feature_flags(HP["features"])
    input_dim = _probe.get_input_dim()

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
    actor.eval()

    # Load and bucket instances
    instance_files = sorted(glob2.glob(os.path.join(args.DATA_PATH, "*.vrp")))
    if not instance_files:
        print(f"No .vrp files found in {args.DATA_PATH}")
        return

    instances = [load_instance_data(f) for f in instance_files]
    print(f"Loaded {len(instances)} instances.")

    buckets = create_buckets(instances, args.mode, args.n_buckets)
    print(f"Mode: {args.mode} | Buckets: {len(buckets)}")

    all_results = []
    for bucket_id, bucket in enumerate(tqdm(buckets, desc="Buckets")):
        max_N = max(inst["n_nodes"] for inst in bucket)
        n_range = (min(inst["n_nodes"] for inst in bucket), max_N)
        print(f"\nBucket {bucket_id}: {len(bucket)} instances, N in {n_range}, padded to {max_N}")

        bucket_results = solve_bucket(bucket, bucket_id, max_N, actor, HP, args)
        all_results.extend(bucket_results)

        for res in bucket_results:
            gap_str = f"{res['Gap_Percent']:.2f}%" if not np.isnan(res["Gap_Percent"]) else "N/A"
            tqdm.write(
                f"  {res['Instance']} (N={res['Nodes']}): "
                f"Opt={res['Optimal_Cost']} | LGSA={res['LGSA_Cost']} | Gap={gap_str}"
            )

    df = pd.DataFrame(all_results)
    out_file = os.path.join(base_path, "cvrplib_X_batch_results.csv")
    df.to_csv(out_file, index=False)

    print("\n--- Summary ---")
    print(f"Results saved to {out_file}")
    valid_gaps = df["Gap_Percent"].dropna()
    if not valid_gaps.empty:
        print(f"Average Gap: {valid_gaps.mean():.2f}%")


if __name__ == "__main__":
    main()
