"""
Multistart study: one fixed instance, many random initialisations.

Takes a *single* CVRP instance, replicates it ``--n_starts`` times into one
batch, gives every copy a different random initial solution, and runs the
LG-SA search once (all restarts in parallel on one GPU). This isolates the
solver's own variance --- the instance is held fixed, so the spread of final
costs is due only to the random initialisation and the stochastic (sampling +
Metropolis) search, not to instance-to-instance difficulty.

It answers two practical questions: how much does the final solution quality
depend on the luck of the initialisation, and how much does a multistart
"best-of-K" buy over a single run.

Reuses eval/solver.py; nothing in src/ is modified.

Usage
-----
    uv run eval/bench_multistart.py --FOLDER BEST --dim 100 \
        --n_starts 10000 --OUTER_STEPS 10000

    # a different instance from the test set, half precision
    uv run eval/bench_multistart.py --FOLDER BEST --instance_idx 42 \
        --dtype float16
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from eval_io import find_model, init_problem_parameters
from solver import build_actor, build_problem, run_lgsa, set_seed, warmup_cuda

from init import initialize_test_problem

_DTYPE_MAP = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}

# Okabe-Ito, matching eval/trace_curve.py (CVD-safe, greyscale-legible).
C_FINAL = "#0072B2"  # blue  -- LG-SA final costs
C_INIT = "#D55E00"  # vermillion -- initial (constructive) costs


# ============================================================================
# PLOT
# ============================================================================


def make_plot(
    init_costs: np.ndarray,
    final_costs: np.ndarray,
    out_path: str,
    *,
    instance_id: str,
    n_steps: int,
) -> None:
    """Two panels: the wide initial-cost distribution and the narrow final one,
    on their own x-scales, so the contraction produced by the search is visible."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 300,
        }
    )

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(6.6, 2.4))

    for ax in (ax0, ax1):
        ax.grid(True, which="major", linewidth=0.4, color="0.9", zorder=0)
        ax.set_axisbelow(True)
        ax.set_ylabel("Count")

    # --- (a) initial constructive solutions ---
    ax0.hist(init_costs, bins=60, color=C_INIT, alpha=0.85, zorder=2)
    ax0.axvline(init_costs.mean(), color="0.2", lw=1.0, ls=(0, (4, 2)), zorder=3)
    ax0.set_xlabel("Initial cost")
    ax0.set_title(f"(a) random inits  (mean {init_costs.mean():.1f})", loc="left")

    # --- (b) LG-SA final solutions ---
    best = final_costs.min()
    mean = final_costs.mean()
    ax1.hist(final_costs, bins=60, color=C_FINAL, alpha=0.85, zorder=2)
    ax1.axvline(best, color="0.2", lw=1.1, zorder=4)
    ax1.axvline(mean, color="0.2", lw=1.0, ls=(0, (4, 2)), zorder=4)
    ax1.annotate(
        f"best {best:.2f}",
        xy=(best, 1.0),
        xycoords=("data", "axes fraction"),
        xytext=(3, -2),
        textcoords="offset points",
        va="top",
        ha="left",
        fontsize=6.5,
        color="0.35",
    )
    ax1.annotate(
        f"mean {mean:.2f}",
        xy=(mean, 1.0),
        xycoords=("data", "axes fraction"),
        xytext=(3, -2),
        textcoords="offset points",
        va="top",
        ha="left",
        fontsize=6.5,
        color="0.35",
    )
    ax1.set_xlabel("LG-SA final cost")
    ax1.set_title(f"(b) after {n_steps:,} steps", loc="left")

    fig.suptitle(
        f"{len(final_costs):,} restarts of instance {instance_id}", fontsize=9, y=1.02
    )
    fig.tight_layout(pad=0.4)
    for ext in ("pdf", "png"):
        fig.savefig(f"{out_path}.{ext}", bbox_inches="tight")
    plt.close(fig)


# ============================================================================
# CLI
# ============================================================================


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Multistart variance study: one instance, many random inits, one batch."
    )
    p.add_argument("--FOLDER", type=str, default="BEST", help="Model folder under wandb/LGSA/")
    p.add_argument("--dim", type=int, default=100, choices=[10, 20, 50, 100, 200, 500, 1000])
    p.add_argument("--DATA", type=str, default="nazari", choices=["nazari", "uchoa"])
    p.add_argument(
        "--DATA_SOURCE",
        type=str,
        default="neuopt",
        choices=["default", "neuopt"],
        help="nazari test set: 'default' (gen_nazari_{dim}.pt) or 'neuopt' (cvrp_{dim}.pkl)",
    )
    p.add_argument(
        "--instance_idx", type=int, default=0, help="Which test-set instance to replicate"
    )
    p.add_argument(
        "--n_starts", type=int, default=10000, help="Number of random restarts (batch size)"
    )
    p.add_argument("--OUTER_STEPS", type=int, default=10000)
    p.add_argument(
        "--INIT",
        type=str,
        default="random",
        choices=["random", "isolate", "sweep", "nearest_neighbor", "Clark_and_Wright"],
    )
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument(
        "--device",
        type=str,
        default=(
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        ),
    )
    p.add_argument("--dtype", type=str, default="float32", choices=list(_DTYPE_MAP.keys()))
    p.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output basename (default: res/<FOLDER>/multistart_<dim>_<idx>)",
    )
    return p


def _summary(name: str, x: np.ndarray) -> dict:
    q = np.percentile(x, [0, 5, 25, 50, 75, 95, 100])
    return {
        "dist": name,
        "mean": x.mean(),
        "std": x.std(),
        "cv_pct": 100.0 * x.std() / x.mean(),
        "min": q[0],
        "p5": q[1],
        "p25": q[2],
        "median": q[3],
        "p75": q[4],
        "p95": q[5],
        "max": q[6],
        "iqr": q[4] - q[2],
        "range": q[6] - q[0],
    }


def main() -> None:
    args = build_parser().parse_args()
    dtype = _DTYPE_MAP[args.dtype]
    set_seed(args.seed)

    model_path = find_model(args.FOLDER)
    print(f"Model: {model_path}")

    cfg = {
        "PROBLEM_DIM": args.dim,
        "N_PROBLEMS": args.n_starts,
        "OUTER_STEPS": args.OUTER_STEPS,
        "DEVICE": args.device,
        "SEED": args.seed,
        "LOAD_PB": True,
        "INIT": args.INIT,
        "MULTI_INIT": False,
        "DATA": args.DATA,
        "BASELINE": False,
    }
    HP = init_problem_parameters(model_path, cfg)

    # --- Load one instance and replicate it into a batch of n_starts copies. ---
    # We ask the loader for enough instances to reach instance_idx, then keep one.
    src_problem, _ = initialize_test_problem(
        config=HP,
        test_dim=args.dim,
        n_test_problems=args.instance_idx + 1,
        init_method=args.INIT,
        data=args.DATA,
        device=args.device,
        source=args.DATA_SOURCE,
    )
    idx = args.instance_idx
    coords1 = src_problem.coords[idx : idx + 1]  # [1, N, 2]
    demands1 = src_problem.demands[idx : idx + 1]  # [1, N]
    cap1 = src_problem.capacity[idx : idx + 1]  # [1, 1]

    coords = coords1.repeat(args.n_starts, 1, 1).contiguous()
    demands = demands1.repeat(args.n_starts, 1).contiguous()
    capacity = cap1.repeat(args.n_starts, 1).contiguous()

    problem = build_problem(HP, dim=args.dim, n_problems=args.n_starts, device=args.device)
    problem.generate_params(coords, demands, capacity)

    # Each row gets an independent random initial solution (random_init_batch
    # shuffles the client order per batch element), so the instance is fixed but
    # the starting points differ.
    init_x = problem.generate_init_state(args.INIT, False)
    init_costs = problem.cost(init_x).float().cpu().numpy()

    input_dim = problem.get_input_dim()
    actor = build_actor(
        HP, model_path, input_dim, device=args.device, seed=args.seed, dtype=dtype
    )
    warmup_cuda()

    instance_id = str(idx)
    print(
        f"Instance {instance_id}: N={args.dim}, "
        f"{args.n_starts} random restarts, init cost {init_costs.mean():.4f} "
        f"(min {init_costs.min():.4f}, max {init_costs.max():.4f})"
    )

    t0 = time.time()
    result = run_lgsa(actor, problem, init_x, HP, outer_steps=args.OUTER_STEPS, dtype=dtype)
    elapsed = time.time() - t0

    final_costs = problem.cost(result["best_x"].to(args.device)).float().cpu().numpy()
    print(f"Done in {elapsed:.1f}s ({elapsed / args.n_starts * 1e3:.3f} ms/restart)")

    # --- Numbers -------------------------------------------------------------
    best = float(final_costs.min())
    summ = pd.DataFrame([_summary("init", init_costs), _summary("final", final_costs)])

    # Gap of a single random run (the mean) over the best-of-K, and of the worst.
    gap_mean = 100.0 * (final_costs.mean() - best) / best
    gap_worst = 100.0 * (final_costs.max() - best) / best
    # Expected best-of-k as a function of restart count (sub-sampling curve).
    ks = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, len(final_costs)]
    ks = [k for k in ks if k <= len(final_costs)]
    rng = np.random.default_rng(0)
    bok = {}
    for k in ks:
        # mean over 200 random subsets of the min of k restarts
        picks = rng.integers(0, len(final_costs), size=(200, k))
        bok[k] = float(final_costs[picks].min(axis=1).mean())

    print("\n--- Distribution summary ---")
    with pd.option_context("display.float_format", lambda v: f"{v:.4f}"):
        print(summ.to_string(index=False))
    print(
        f"\nSingle run (mean) is {gap_mean:.2f}% above best-of-{len(final_costs)}; "
        f"worst run {gap_worst:.2f}% above."
    )
    print("Expected best-of-k cost:")
    for k in ks:
        print(f"  k={k:>6}: {bok[k]:.4f}  ({100 * (bok[k] - best) / best:+.2f}% over best-of-all)")

    # --- Save ----------------------------------------------------------------
    out = args.out or os.path.join("res", args.FOLDER, f"multistart_{args.dim}_{idx}")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    pd.DataFrame(
        {"restart": np.arange(len(final_costs)), "init_cost": init_costs, "final_cost": final_costs}
    ).to_csv(f"{out}.csv", index=False)
    summ.to_csv(f"{out}_summary.csv", index=False)
    pd.DataFrame({"k": ks, "best_of_k": [bok[k] for k in ks]}).to_csv(f"{out}_bestofk.csv", index=False)

    make_plot(init_costs, final_costs, out, instance_id=instance_id, n_steps=args.OUTER_STEPS)

    print(f"\nPer-restart CSV -> {out}.csv")
    print(f"Summary CSV     -> {out}_summary.csv")
    print(f"Best-of-k CSV   -> {out}_bestofk.csv")
    print(f"Figure          -> {out}.pdf / {out}.png")


if __name__ == "__main__":
    main()
