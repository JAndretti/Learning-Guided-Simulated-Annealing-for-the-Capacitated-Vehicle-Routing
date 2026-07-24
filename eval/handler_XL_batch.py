import argparse
import csv
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

# Reuse the padding / bucketing helpers from the X batch handler — the mechanics
# (ghost depot nodes, percentile buckets) are identical; only the cost convention,
# memory-aware batch sizing, external BKS reference and blind-SA pass differ here.
from handler_X_batch import _create_buckets, _pad_to_N


def add_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--DATA_PATH", type=str, default="bdd/XL")
    parser.add_argument(
        "--BKS_PATH", type=str, default="bdd/XL_bks.csv",
        help="CSV with columns Instance,Initial_BKS,Final_BKS (from Queiroga et al. 2026, Table 1)."
    )
    parser.add_argument(
        "--bks_ref", type=str, default="initial", choices=["initial", "final"],
        help="Which BKS column to compute the gap against. 'initial' matches the paper's Table 2 reference."
    )
    parser.add_argument(
        "--mode", type=str, default="bucket", choices=["bucket", "global"],
        help="'bucket': split by percentile groups; 'global': one batch for all."
    )
    parser.add_argument("--n_buckets", type=int, default=20)
    parser.add_argument(
        "--mem_cap", type=float, default=5.0e7,
        help="Memory budget proxy: within a bucket, batch_size = max(1, floor(mem_cap / max_N^2)). "
             "Caps B * max_N^2 so large-N buckets fall to batch 1 and small-N buckets batch many. "
             "Default 5e7 is float32-safe on ~12 GB (batch 1 at N>~7000); raise for float16."
    )
    parser.add_argument(
        "--no-baseline", dest="BASELINE", action="store_false", default=True,
        help="Skip the blind-SA (baseline) pass; run LG-SA only."
    )


def _load_bks(path: str, key: str) -> dict[str, float]:
    col = "Initial_BKS" if key == "initial" else "Final_BKS"
    bks: dict[str, float] = {}
    if not os.path.exists(path):
        print(f"WARNING: BKS file {path} not found; gaps will be NaN.")
        return bks
    with open(path) as fh:
        for row in csv.DictReader(fh):
            bks[row["Instance"]] = float(row[col])
    return bks


def _batch_size_for(max_N: int, mem_cap: float) -> int:
    return max(1, int(mem_cap // (max_N ** 2)))


def _solve_bucket(
    bucket: list[dict],
    bucket_id: int,
    max_N: int,
    actor,
    HP: dict,
    args: argparse.Namespace,
    bks: dict[str, float],
) -> list[dict]:
    results = []
    bs = _batch_size_for(max_N, args.mem_cap)

    for start in range(0, len(bucket), bs):
        mini = bucket[start : start + bs]
        B = len(mini)

        coords, demands, capacities = _pad_to_N(mini, max_N, args.device)

        problem = build_problem(HP, dim=max_N - 1, n_problems=B, device=args.device)
        problem.generate_params(coords, demands, capacities)
        init_x = problem.generate_init_state(args.INIT, False)

        # Compact ghost nodes out of init_x so SA never targets them (same as X_batch).
        for i, inst in enumerate(mini):
            actual_N = inst["n_nodes"]
            sol = init_x[i, :, 0]
            is_real = (sol == 0) | (sol < actual_N)
            real_part = sol[is_real]
            n_ghost = sol.shape[0] - real_part.shape[0]
            if n_ghost > 0:
                padding = torch.zeros(n_ghost, dtype=sol.dtype, device=sol.device)
                init_x[i, :, 0] = torch.cat([real_part, padding])
        problem.init_parameters(init_x)

        # --- Blind SA (baseline) then LG-SA, from the SAME initial solution ---
        # Both run_lgsa calls re-init from their passed init_x internally; the distance
        # matrix (coords-derived) is unchanged between passes, so no re-generate needed.
        sa_best = None
        sa_elapsed = float("nan")
        if args.BASELINE:
            t0 = time.time()
            sa_res = run_lgsa(
                actor, problem, init_x.clone(), HP,
                outer_steps=args.OUTER_STEPS, baseline=True, dtype=args.torch_dtype,
            )
            sa_elapsed = time.time() - t0
            sa_best = sa_res["best_x"].cpu().numpy()

        t0 = time.time()
        lg_res = run_lgsa(
            actor, problem, init_x.clone(), HP,
            outer_steps=args.OUTER_STEPS, baseline=False, dtype=args.torch_dtype,
        )
        lg_elapsed = time.time() - t0
        lg_best = lg_res["best_x"].cpu().numpy()

        for i, inst in enumerate(mini):
            n = inst["n_nodes"]
            lg_cost = extract_and_cost(lg_best[i, :, 0].tolist(), n, inst["raw_coords"], rounded=True)
            ref = bks.get(inst["name"], float("nan"))
            lg_gap = 100 * (lg_cost - ref) / ref if not np.isnan(ref) else float("nan")

            if args.BASELINE:
                sa_cost = extract_and_cost(sa_best[i, :, 0].tolist(), n, inst["raw_coords"], rounded=True)
                sa_gap = 100 * (sa_cost - ref) / ref if not np.isnan(ref) else float("nan")
            else:
                sa_cost = float("nan")
                sa_gap = float("nan")

            results.append({
                "Instance": inst["name"],
                "Nodes": n,
                "BKS": ref,
                "LGSA_Cost": lg_cost,
                "Gap_Percent": lg_gap,
                "SA_Cost": sa_cost,
                "SA_Gap_Percent": sa_gap,
                "Time_sec": lg_elapsed / B,
                "SA_Time_sec": sa_elapsed / B if args.BASELINE else float("nan"),
                "Steps": args.OUTER_STEPS,
                "Bucket_ID": bucket_id,
                "Batch_size_actual": B,
            })

        del problem, init_x
        if args.device == "cuda":
            torch.cuda.empty_cache()

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

    _probe = build_problem(HP, dim=100, n_problems=1, device=args.device)
    input_dim = _probe.get_input_dim()
    actor = build_actor(
        HP, model_path, input_dim, device=args.device, seed=args.seed, dtype=args.torch_dtype
    )

    bks = _load_bks(args.BKS_PATH, args.bks_ref)

    instance_files = sorted(glob2.glob(os.path.join(args.DATA_PATH, "*.vrp")))
    if not instance_files:
        print(f"No .vrp files found in {args.DATA_PATH}")
        return

    # XL has no .sol files; the reference comes from --BKS_PATH.
    instances = [load_vrp_instance(f, load_solution=False) for f in instance_files]
    print(f"Loaded {len(instances)} instances. BKS ref: {args.bks_ref} ({len(bks)} entries).")

    buckets = _create_buckets(instances, args.mode, args.n_buckets)
    print(f"Mode: {args.mode} | Buckets: {len(buckets)} | baseline: {args.BASELINE}")

    warmup_cuda()

    all_results = []
    for bucket_id, bucket in enumerate(tqdm(buckets, desc="Buckets")):
        max_N = max(inst["n_nodes"] for inst in bucket)
        n_range = (min(inst["n_nodes"] for inst in bucket), max_N)
        bs = _batch_size_for(max_N, args.mem_cap)
        print(f"\nBucket {bucket_id}: {len(bucket)} instances, N in {n_range}, padded to {max_N}, batch {bs}")

        bucket_results = _solve_bucket(bucket, bucket_id, max_N, actor, HP, args, bks)
        all_results.extend(bucket_results)
        if args.device == "cuda":
            torch.cuda.empty_cache()

        for res in bucket_results:
            gap_str = f"{res['Gap_Percent']:.2f}%" if not np.isnan(res["Gap_Percent"]) else "N/A"
            sa_str = f" | SA={res['SA_Cost']}" if args.BASELINE else ""
            tqdm.write(
                f"  {res['Instance']} (N={res['Nodes']}): "
                f"BKS={res['BKS']:.0f} | LGSA={res['LGSA_Cost']} | Gap={gap_str}{sa_str}"
            )

    df = pd.DataFrame(all_results)

    # Paper split: 50 smaller / 50 larger by N (Queiroga et al. 2026, Table 2).
    n_sorted = df.sort_values("Nodes")
    half = len(n_sorted) // 2
    small_names = set(n_sorted.iloc[:half]["Instance"])

    def _mean_gap(col: str, subset: pd.DataFrame) -> float:
        v = subset[col].dropna()
        return float(v.mean()) if not v.empty else float("nan")

    small = df[df["Instance"].isin(small_names)]
    large = df[~df["Instance"].isin(small_names)]
    summary_rows = pd.DataFrame([
        {"Instance": "LGSA avg gap (%)",         "Gap_Percent": _mean_gap("Gap_Percent", df)},
        {"Instance": "LGSA avg gap (%) small",   "Gap_Percent": _mean_gap("Gap_Percent", small)},
        {"Instance": "LGSA avg gap (%) large",   "Gap_Percent": _mean_gap("Gap_Percent", large)},
        {"Instance": "SA avg gap (%)",           "Gap_Percent": _mean_gap("SA_Gap_Percent", df)},
        {"Instance": "SA avg gap (%) small",     "Gap_Percent": _mean_gap("SA_Gap_Percent", small)},
        {"Instance": "SA avg gap (%) large",     "Gap_Percent": _mean_gap("SA_Gap_Percent", large)},
    ])
    df_out = pd.concat([df, summary_rows], ignore_index=True)

    base_path = f"res/{args.FOLDER}"
    out = save_results(df_out, base_path, "cvrplib_XL_batch_results")
    print("\n--- Summary ---")
    print(f"Results saved to {out}")
    print(f"LGSA avg gap:        {_mean_gap('Gap_Percent', df):.2f}%  "
          f"(small {_mean_gap('Gap_Percent', small):.2f}%, large {_mean_gap('Gap_Percent', large):.2f}%)")
    if args.BASELINE:
        print(f"Blind-SA avg gap:    {_mean_gap('SA_Gap_Percent', df):.2f}%  "
              f"(small {_mean_gap('SA_Gap_Percent', small):.2f}%, large {_mean_gap('SA_Gap_Percent', large):.2f}%)")
