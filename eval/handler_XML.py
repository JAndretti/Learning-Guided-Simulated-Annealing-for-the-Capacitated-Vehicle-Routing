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

from costs import exact_euclidean_cost
from eval_io import find_model, init_problem_parameters, load_vrp_instance, save_results
from solver import build_actor, build_problem, run_lgsa, set_seed, warmup_cuda


def add_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--DATA_PATH", type=str, default="bdd/XML", help="Path to Queiroga XML .vrp files"
    )
    parser.add_argument(
        "--SOL_PATH", type=str, default="bdd/solutions", help="Directory containing .sol files"
    )


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
        "DATA": "queiroga_xml",
        "BASELINE": False,
    }
    HP = init_problem_parameters(model_path, cfg)

    instance_files = sorted(glob2.glob(os.path.join(args.DATA_PATH, "*.vrp")))
    if not instance_files:
        print(f"No .vrp files found in {args.DATA_PATH}")
        return

    print(f"Loading {len(instance_files)} instances...")
    instances = [
        load_vrp_instance(f, load_solution=True, sol_dir=args.SOL_PATH)
        for f in tqdm(instance_files, desc="Loading")
    ]

    # All XML instances must be the same size; verify and filter
    first_n = instances[0]["n_nodes"]
    instances = [d for d in instances if d["n_nodes"] == first_n]
    if len(instances) < len(instance_files):
        print(f"Filtered to {len(instances)} instances with N={first_n}")

    B = len(instances)
    N = first_n

    # Stack into batch tensors [B, N, 2], [B, N], [B, 1]
    batch_coords = torch.stack([d["norm_coords"] for d in instances]).to(args.device)
    batch_demands = torch.stack(
        [torch.tensor(d["raw_demand"]) for d in instances]
    ).to(args.device)
    batch_caps = (
        torch.tensor([d["capacity"] for d in instances], dtype=torch.float32)
        .unsqueeze(-1)
        .to(args.device)
    )

    problem = build_problem(HP, dim=N - 1, n_problems=B, device=args.device)
    input_dim = problem.get_input_dim()
    actor = build_actor(HP, model_path, input_dim, device=args.device, seed=args.seed)

    problem.generate_params(batch_coords, batch_demands, batch_caps)
    init_x = problem.generate_init_state(args.INIT, False)

    warmup_cuda()
    print(f"Running LGSA on batch of {B} instances (N={N})...")

    t0 = time.time()
    result = run_lgsa(actor, problem, init_x, HP, outer_steps=args.OUTER_STEPS)
    total_elapsed = time.time() - t0
    print(f"Done in {total_elapsed:.2f}s ({total_elapsed / B:.4f}s/instance)")

    best_x = result["best_x"].cpu().numpy()  # [B, seq_len, 1]

    rows = []
    for i, data in enumerate(instances):
        sol = best_x[i, :, 0].tolist()
        cost = exact_euclidean_cost(sol, data["raw_coords"])
        opt = data["optimal_cost"]
        gap = 100 * (cost - opt) / opt if not np.isnan(opt) else float("nan")
        rows.append({
            "Instance": data["name"],
            "Nodes": N,
            "Optimal_Cost": opt,
            "LGSA_Cost": cost,
            "Gap_Percent": gap,
            "Time_sec": total_elapsed / B,
        })

    df = pd.DataFrame(rows)
    base_path = f"res/{args.FOLDER}"
    out = save_results(df, base_path, "queiroga_xml_results")

    print("\n--- Summary (first 10) ---")
    print(df[["Instance", "Optimal_Cost", "LGSA_Cost", "Gap_Percent"]].head(10).to_string())
    print(f"\nResults saved to {out}")
    valid_gaps = df["Gap_Percent"].dropna()
    if not valid_gaps.empty:
        print(f"Average Gap: {valid_gaps.mean():.2f}%")
