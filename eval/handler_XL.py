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

from costs import cvrplib_rounded_cost
from eval_io import find_model, init_problem_parameters, load_vrp_instance, save_results
from solver import build_actor, build_problem, run_lgsa, set_seed


def add_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--DATA_PATH", type=str, default="bdd/XL", help="Path to CVRPLib Set XL .vrp files"
    )


def _solve(data: dict, actor, HP: dict, args: argparse.Namespace) -> dict:
    n = data["n_nodes"]
    problem = build_problem(HP, dim=n - 1, n_problems=1, device=args.device)
    problem.generate_params(
        data["norm_coords"].unsqueeze(0).to(args.device),
        torch.tensor(data["raw_demand"]).unsqueeze(0).to(args.device),
        torch.tensor([data["capacity"]], dtype=torch.float32).unsqueeze(-1).to(args.device),
    )
    init_x = problem.generate_init_state(args.INIT, False)

    t0 = time.time()
    result = run_lgsa(actor, problem, init_x, HP, outer_steps=args.OUTER_STEPS)
    elapsed = time.time() - t0

    sol = result["best_x"][0, :, 0].cpu().tolist()
    cost = cvrplib_rounded_cost(sol, data["raw_coords"])

    return {
        "Instance": data["name"],
        "Nodes": n,
        "LGSA_Cost": cost,
        "Time_sec": elapsed,
        "Steps": args.OUTER_STEPS,
    }


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

    _probe = build_problem(HP, dim=100, n_problems=1, device=args.device)
    input_dim = _probe.get_input_dim()
    actor = build_actor(HP, model_path, input_dim, device=args.device, seed=args.seed)

    # load_solution=False: XL has no .sol files
    instance_files = sorted(glob2.glob(os.path.join(args.DATA_PATH, "*.vrp")))
    if not instance_files:
        print(f"No .vrp files found in {args.DATA_PATH}")
        return
    print(f"Found {len(instance_files)} instances.")

    # Full solve on middle instance warms up both CUDA and JIT compilation
    print("Warming up...")
    mid_data = load_vrp_instance(instance_files[len(instance_files) // 2], load_solution=False)
    _solve(mid_data, actor, HP, args)

    results = []
    for filepath in tqdm(instance_files):
        try:
            data = load_vrp_instance(filepath, load_solution=False)
            row = _solve(data, actor, HP, args)
            results.append(row)
            tqdm.write(
                f"{row['Instance']} (N={row['Nodes']}): "
                f"LGSA_Cost={row['LGSA_Cost']} | Time={row['Time_sec']:.2f}s"
            )
        except Exception as e:
            import traceback
            print(f"Error processing {filepath}: {e}")
            traceback.print_exc()

    df = pd.DataFrame(results)
    base_path = f"res/{args.FOLDER}"
    out = save_results(df, base_path, "cvrplib_XL_results")
    print(f"\nResults saved to {out}")
    if not df.empty:
        print(f"Average LGSA Cost: {df['LGSA_Cost'].mean():.2f}")
        print(f"Average Time per Instance: {df['Time_sec'].mean():.2f}s")
