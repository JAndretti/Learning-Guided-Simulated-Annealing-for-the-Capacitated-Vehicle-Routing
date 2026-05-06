import argparse
import os
import sys
import time

import glob2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from costs import extract_and_cost
from eval_io import find_model, init_problem_parameters, load_vrp_instance, save_results
from solver import build_actor, build_problem, run_lgsa, set_seed, warmup_cuda


def add_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--DATA_PATH", type=str, default="bdd/X")
    parser.add_argument(
        "--mode", type=str, default="bucket", choices=["bucket", "global"],
        help="'bucket': split by percentile groups; 'global': one batch for all"
    )
    parser.add_argument("--n_buckets", type=int, default=5)
    parser.add_argument(
        "--batch_size", type=str, default="all",
        help="Mini-batch size within a bucket. 'all' uses the full bucket at once."
    )


def _create_buckets(instances: list[dict], mode: str, n_buckets: int) -> list[list[dict]]:
    sorted_inst = sorted(instances, key=lambda x: x["n_nodes"])
    if mode == "global":
        return [sorted_inst]
    buckets = np.array_split(sorted_inst, n_buckets)
    return [list(b) for b in buckets if len(b) > 0]


def _pad_to_N(instances: list[dict], max_N: int, device: str):
    """
    Pad each instance's coords/demands to max_N with ghost depot nodes.

    Ghost nodes have coord = depot coord and demand = 0. The CVRP treats them
    as extra depots, so they are never selected as move sources.

    Returns:
        coords:     [B, max_N, 2]
        demands:    [B, max_N]
        capacities: [B, 1]
    """
    coords_list, demands_list, caps_list = [], [], []
    for inst in instances:
        nc = inst["norm_coords"]  # [N_i, 2]
        N_i = nc.shape[0]
        pad = max_N - N_i
        if pad > 0:
            nc = torch.cat([nc, nc[0:1].expand(pad, -1)], dim=0)
        dem = torch.tensor(inst["raw_demand"])
        if pad > 0:
            dem = torch.cat([dem, torch.zeros(pad, dtype=dem.dtype)], dim=0)
        coords_list.append(nc)
        demands_list.append(dem)
        caps_list.append(inst["capacity"])
    coords = torch.stack(coords_list).to(device)
    demands = torch.stack(demands_list).to(device)
    capacities = torch.tensor(caps_list, dtype=torch.float32).unsqueeze(-1).to(device)
    return coords, demands, capacities


def _solve_bucket(
    bucket: list[dict],
    bucket_id: int,
    max_N: int,
    actor,
    HP: dict,
    args: argparse.Namespace,
) -> list[dict]:
    results = []
    bs = len(bucket) if args.batch_size == "all" else int(args.batch_size)

    for start in range(0, len(bucket), bs):
        mini = bucket[start : start + bs]
        B = len(mini)

        coords, demands, capacities = _pad_to_N(mini, max_N, args.device)

        problem = build_problem(HP, dim=max_N - 1, n_problems=B, device=args.device)
        problem.generate_params(coords, demands, capacities)
        init_x = problem.generate_init_state(args.INIT, False)

        # Compact ghost nodes out of init_x so SA never targets them
        for i, inst in enumerate(mini):
            actual_N = inst["n_nodes"]
            sol = init_x[i, :, 0]
            is_real = (sol == 0) | (sol < actual_N)
            real_part = sol[is_real]
            n_ghost = sol.shape[0] - real_part.shape[0]
            if n_ghost > 0:
                padding = torch.zeros(n_ghost, dtype=sol.dtype, device=sol.device)
                init_x[i, :, 0] = torch.cat([real_part, padding])
        # Explicit init here so the compacted init_x (ghost nodes pushed to the end)
        # is the state SA starts from. inf_test_model calls init_parameters again
        # internally, but it receives the same tensor so the result is identical.
        problem.init_parameters(init_x)

        t0 = time.time()
        result = run_lgsa(actor, problem, init_x, HP, outer_steps=args.OUTER_STEPS)
        elapsed = time.time() - t0

        best_x = result["best_x"].cpu().numpy()  # [B, seq_len, 1]

        for i, inst in enumerate(mini):
            sol = best_x[i, :, 0].tolist()
            cost = extract_and_cost(sol, inst["n_nodes"], inst["raw_coords"], rounded=True)
            opt = inst["optimal_cost"]
            gap = 100 * (cost - opt) / opt if not np.isnan(opt) else float("nan")
            results.append({
                "Instance": inst["name"],
                "Nodes": inst["n_nodes"],
                "Optimal_Cost": opt,
                "LGSA_Cost": cost,
                "Gap_Percent": gap,
                "Time_sec": elapsed / B,
                "Steps": args.OUTER_STEPS,
                "Bucket_ID": bucket_id,
                "Batch_size_actual": B,
            })

    return results


def run(args: argparse.Namespace) -> None:
    set_seed(args.seed)

    model_path = find_model(args.FOLDER)
    print(f"Evaluating model: {model_path}")

    cfg = {
        "PROBLEM_DIM": 100,
        "DEVICE": args.device,
        "SEED": args.seed,
        "LOAD_PB": True,
        "INIT": args.INIT,
        "OUTER_STEPS": args.OUTER_STEPS,
        "DATA": "cvrplib",
        "BASELINE": False,
    }
    HP = init_problem_parameters(model_path, cfg)

    # Build actor once — input_dim depends only on feature flags
    _probe = build_problem(HP, dim=100, n_problems=1, device=args.device)
    input_dim = _probe.get_input_dim()
    actor = build_actor(HP, model_path, input_dim, device=args.device, seed=args.seed)

    instance_files = sorted(glob2.glob(os.path.join(args.DATA_PATH, "*.vrp")))
    if not instance_files:
        print(f"No .vrp files found in {args.DATA_PATH}")
        return

    instances = [load_vrp_instance(f) for f in instance_files]
    print(f"Loaded {len(instances)} instances.")

    buckets = _create_buckets(instances, args.mode, args.n_buckets)
    print(f"Mode: {args.mode} | Buckets: {len(buckets)}")

    warmup_cuda()

    all_results = []
    for bucket_id, bucket in enumerate(tqdm(buckets, desc="Buckets")):
        max_N = max(inst["n_nodes"] for inst in bucket)
        n_range = (min(inst["n_nodes"] for inst in bucket), max_N)
        print(f"\nBucket {bucket_id}: {len(bucket)} instances, N in {n_range}, padded to {max_N}")

        bucket_results = _solve_bucket(bucket, bucket_id, max_N, actor, HP, args)
        all_results.extend(bucket_results)

        for res in bucket_results:
            gap_str = f"{res['Gap_Percent']:.2f}%" if not np.isnan(res["Gap_Percent"]) else "N/A"
            tqdm.write(
                f"  {res['Instance']} (N={res['Nodes']}): "
                f"Opt={res['Optimal_Cost']} | LGSA={res['LGSA_Cost']} | Gap={gap_str}"
            )

    df = pd.DataFrame(all_results)

    valid_gaps = df["Gap_Percent"].dropna()
    gap_small = df.loc[df["Nodes"] < 330, "Gap_Percent"].dropna()
    gap_large = df.loc[df["Nodes"] > 330, "Gap_Percent"].dropna()

    summary_rows = pd.DataFrame([
        {"Instance": "Avg. gap (%)",         "Gap_Percent": valid_gaps.mean() if not valid_gaps.empty else float("nan")},
        {"Instance": "Avg. gap (%) (n<330)", "Gap_Percent": gap_small.mean()  if not gap_small.empty  else float("nan")},
        {"Instance": "Avg. gap (%) (n>330)", "Gap_Percent": gap_large.mean()  if not gap_large.empty  else float("nan")},
    ])
    df = pd.concat([df, summary_rows], ignore_index=True)

    base_path = f"res/{args.FOLDER}"
    out = save_results(df, base_path, "cvrplib_X_batch_results")
    print("\n--- Summary ---")
    print(f"Results saved to {out}")
    if not valid_gaps.empty:
        print(f"Average Gap:        {valid_gaps.mean():.2f}%")
    if not gap_small.empty:
        print(f"Average Gap (n<330): {gap_small.mean():.2f}%")
    if not gap_large.empty:
        print(f"Average Gap (n>330): {gap_large.mean():.2f}%")
