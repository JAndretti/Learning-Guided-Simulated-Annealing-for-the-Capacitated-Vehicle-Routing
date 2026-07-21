"""
Step-by-step cost trace: LG-SA (learned actor) vs. pure SA baseline.

Runs both solvers on the *same* random instances / same initial solutions and
records the batch-mean cost at every SA step, then writes a CSV of the traces
and a publication-ready plot.

Self-contained: the SA loop below mirrors src/sa/sa_test.py but keeps a
per-step trace. Nothing in src/ is modified.

Usage
-----
    # same budget for both
    uv run eval/trace_curve.py --FOLDER Q2_depth --dim 100 --DATA nazari \
        --batch_size 10000 --OUTER_STEPS 10000

    # give the baseline a longer budget (roughly compute-matched)
    uv run eval/trace_curve.py --FOLDER Q2_depth --STEPS_MODEL 10000 \
        --STEPS_BASELINE 50000 --logx
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from eval_io import init_problem_parameters
from solver import build_actor, set_seed, warmup_cuda

from init import initialize_test_problem
from sa.sa_test import metropolis_accept, scale_between, scale_to_unit
from sa.scheduler import Scheduler
from utils import extend_to

_DTYPE_MAP = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}


# ============================================================================
# MODEL LOOKUP
# ============================================================================


def find_best_checkpoint_dir(folder: str) -> tuple[str, str]:
    """
    Search wandb/LGSA/<folder>/ recursively for actor checkpoints named
    ..._loss_<value>.pt and return (model_dir, checkpoint_name) for the lowest
    loss found.

    eval_io.find_model() is not used here: it ranks the *run directories*, whose
    names are timestamps, so every candidate parses to inf and the choice is
    arbitrary. The loss lives in the .pt filenames one level down.

    The returned model_dir is what build_actor() loads from — _load_weights()
    re-scans that directory and re-selects the same lowest-loss file.
    """
    root = os.path.join("wandb", "LGSA", folder)
    if not os.path.isdir(root):
        raise FileNotFoundError(f"No such model folder: {root}")

    best: tuple[float, str] | None = None
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if not (fn.endswith(".pt") and "actor" in fn):
                continue
            try:
                loss = float(fn[:-3].split("_")[-1])
            except ValueError:
                continue
            if best is None or loss < best[0]:
                best = (loss, os.path.join(dirpath, fn))

    if best is None:
        raise FileNotFoundError(f"No actor checkpoints with a parsable loss under {root}")

    ckpt = best[1]
    model_dir = os.path.dirname(ckpt)
    if not os.path.exists(os.path.join(model_dir, "HP.yaml")):
        raise FileNotFoundError(f"No HP.yaml next to checkpoint {ckpt}")
    return model_dir, os.path.basename(ckpt)


# ============================================================================
# TRACED SA LOOP
# ============================================================================


def sa_trace(
    actor,
    problem,
    initial_solution: torch.Tensor,
    config: dict,
    *,
    baseline: bool = False,
    greedy: bool = False,
    dtype: torch.dtype = torch.float32,
    trace_every: int = 1,
    desc: str = "SA",
) -> dict[str, torch.Tensor]:
    """
    SA loop that records the batch-mean current and best-so-far cost every
    `trace_every` steps (index 0 = initial solution).

    Means are accumulated as 0-dim device tensors and stacked once at the end,
    so the loop never syncs to the host.

    Returns dict with keys:
        steps      [T]  step indices of the recorded points
        mean_cur   [T]  mean over the batch of the current cost
        mean_best  [T]  mean over the batch of the best-so-far cost
        best_cost  [B]  final per-instance best cost
    """
    device = str(initial_solution.device)
    total_steps = config["TEST_OUTER_STEPS"]

    scheduler = Scheduler(
        config["SCHEDULER"],
        T_max=config["INIT_TEMP"],
        T_min=config["STOP_TEMP"],
        step_max=total_steps,
    )

    current_solution = initial_solution.clone()
    best_solution = initial_solution.clone()

    current_cost = problem.cost(initial_solution).to(dtype)
    best_cost = current_cost.clone()

    current_temp = torch.tensor([1.0], device=device).repeat(current_cost.shape[0])
    current_temp = scale_between(current_temp, config["STOP_TEMP"], config["INIT_TEMP"]).to(dtype)

    normalized_temp = scale_to_unit(current_temp, config["STOP_TEMP"], config["INIT_TEMP"])
    if baseline:
        current_state = current_solution
    else:
        current_state = (
            problem.to_state(
                *problem.build_state_components(
                    current_solution, normalized_temp, torch.tensor(1.0, device=device)
                )
            )
            .to(device)
            .to(dtype)
        )

    # Trace buffers — float32 means regardless of loop dtype (bf16 means are noisy).
    steps: list[int] = [0]
    trace_cur: list[torch.Tensor] = [current_cost.float().mean()]
    trace_best: list[torch.Tensor] = [best_cost.float().mean()]

    for step in tqdm(range(total_steps), desc=desc, colour="green", leave=False):
        with torch.no_grad():
            if baseline:
                action, _, _, _ = actor.baseline_sample(current_state, problem=problem)
            else:
                action, _, _, _ = actor.sample(current_state, greedy=greedy, problem=problem)

        sol_components, *_ = problem.from_state(current_state)
        proposed_sol, _ = problem.update(sol_components, action)
        proposed_cost = problem.cost(proposed_sol).to(dtype)

        cost_improvement = current_cost - proposed_cost

        if config["METROPOLIS"]:
            is_accepted, _ = metropolis_accept(cost_improvement, current_temp, device)
        else:
            is_accepted = torch.ones_like(cost_improvement)

        current_cost = torch.where(is_accepted.bool(), proposed_cost, current_cost)
        is_accepted_expanded = extend_to(is_accepted, sol_components)
        current_solution = torch.where(
            is_accepted_expanded.bool(), proposed_sol, sol_components
        ).long()

        problem.update_tensor(current_solution)

        is_improvement = (current_cost < best_cost).long()
        best_cost = torch.minimum(current_cost, best_cost)
        is_imp_expanded = extend_to(is_improvement, current_solution)
        best_solution = torch.where(is_imp_expanded.bool(), current_solution, best_solution)

        next_temp = scheduler.step(step).to(device).repeat(current_solution.shape[0]).to(dtype)
        current_temp = next_temp

        adv = torch.tensor(1 - (step / total_steps), device=device)
        model_temp = scale_to_unit(next_temp, config["STOP_TEMP"], config["INIT_TEMP"])
        if baseline:
            next_state = current_solution
        else:
            next_state = (
                problem.to_state(*problem.build_state_components(current_solution, model_temp, adv))
                .to(device)
                .to(dtype)
            )
        current_state = next_state

        if trace_every > 0 and ((step + 1) % trace_every == 0 or step + 1 == total_steps):
            steps.append(step + 1)
            trace_cur.append(current_cost.float().mean())
            trace_best.append(best_cost.float().mean())

    return {
        "steps": torch.tensor(steps),
        "mean_cur": torch.stack(trace_cur).cpu(),
        "mean_best": torch.stack(trace_best).cpu(),
        "best_cost": best_cost.float().cpu(),
        "best_x": best_solution,
    }


# ============================================================================
# PLOT
# ============================================================================

# Okabe-Ito blue / vermillion: CVD-safe (worst adjacent pair ΔE 21.9 protan),
# distinguishable in greyscale print, contrast >= 3:1 on white.
C_MODEL = "#0072B2"
C_BASE = "#D55E00"


LABEL = {"model": "LG-SA", "baseline": "SA"}
COLOUR = {"model": C_MODEL, "baseline": C_BASE}


def make_plot(
    df: pd.DataFrame,
    out_path: str,
    *,
    show_current: bool,
    logx: bool,
    xlabel: str = "SA step",
    vline: float | None = None,
) -> None:
    """
    df is long-format (method, step, mean_best, mean_current) so the two runs may
    have different step budgets and therefore different x-extents.
    """
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

    fig, ax = plt.subplots(figsize=(3.4, 2.4))

    ax.grid(True, which="major", linewidth=0.4, color="0.9", zorder=0)
    ax.set_axisbelow(True)

    # Draw baseline first so the model curve sits on top where they overlap.
    order = [m for m in ("baseline", "model") if m in set(df["method"])]

    if show_current:
        for method in order:
            g = df[df["method"] == method]
            ax.plot(g["step"], g["mean_current"], color=COLOUR[method], lw=0.6, alpha=0.3, zorder=1)

    for z, method in enumerate(order):
        g = df[df["method"] == method]
        ax.plot(
            g["step"],
            g["mean_best"],
            color=COLOUR[method],
            lw=1.4,
            label=LABEL[method],
            zorder=3 + z,
        )

    if logx:
        ax.set_xscale("log")

    # Equal-wall-clock marker: where the baseline has spent as much time as the
    # full model run. Recessive — it annotates the curves, it is not a series.
    if vline is not None:
        ax.axvline(vline, color="0.2", lw=1.1, ls=(0, (4, 2)), zorder=10, alpha=0.95)
        ax.annotate(
            "equal time",
            xy=(vline, 1.0),
            xycoords=("data", "axes fraction"),
            xytext=(3, -2),
            textcoords="offset points",
            va="top",
            ha="left",
            fontsize=6.5,
            color="0.45",
        )

    # Direct labels at each curve's own end — identity is never colour-alone, and
    # the two ends may differ when the step budgets differ.
    x_max = df["step"].max()
    for method in order:
        g = df[df["method"] == method]
        ax.annotate(
            f"{LABEL[method]}  {g['mean_best'].iloc[-1]:.2f}",
            xy=(g["step"].iloc[-1], g["mean_best"].iloc[-1]),
            xytext=(3, 0),
            textcoords="offset points",
            va="center",
            ha="left",
            fontsize=7,
            color="0.25",
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Mean cost")
    ax.margins(x=0.02)
    ax.set_xlim(right=x_max * (1.35 if not logx else 2.6))
    ax.legend(frameon=False, loc="upper right")

    fig.tight_layout(pad=0.3)
    for ext in ("pdf", "png"):
        fig.savefig(f"{out_path}.{ext}", bbox_inches="tight")
    plt.close(fig)


# ============================================================================
# CLI
# ============================================================================


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Step-by-step mean-cost comparison: LG-SA vs SA baseline"
    )
    # --- shared with eval/run.py ---
    p.add_argument("--FOLDER", type=str, default="BEST", help="Model folder under wandb/LGSA/")
    p.add_argument(
        "--INIT",
        type=str,
        default="random",
        choices=["random", "isolate", "sweep", "nearest_neighbor", "Clark_and_Wright"],
    )
    p.add_argument(
        "--OUTER_STEPS", type=int, default=10000, help="Step budget for both runs unless overridden"
    )
    p.add_argument(
        "--STEPS_MODEL",
        type=int,
        default=None,
        help="Step budget for LG-SA (default: --OUTER_STEPS)",
    )
    p.add_argument(
        "--STEPS_BASELINE",
        type=int,
        default=None,
        help="Step budget for the SA baseline (default: --OUTER_STEPS)",
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
    # --- shared with eval/handler_random.py ---
    p.add_argument("--dim", type=int, default=100, choices=[10, 20, 50, 100, 500, 1000])
    p.add_argument("--DATA", type=str, default="nazari", choices=["nazari", "uchoa"])
    p.add_argument("--batch_size", type=int, default=10000)
    p.add_argument("--greedy", dest="GREEDY", action="store_true", default=False)
    p.add_argument("--no-metro", dest="METROPOLIS", action="store_false", default=True)
    # --- trace-specific ---
    p.add_argument("--trace_every", type=int, default=10, help="Record the mean cost every k steps")
    p.add_argument(
        "--x_frac",
        action="store_true",
        help="Plot the x axis as fraction of each run's budget instead of absolute steps "
        "(use when the two budgets differ and you want to compare anneal shape)",
    )
    p.add_argument("--logx", action="store_true", help="Log-scale the step axis")
    p.add_argument(
        "--no-equal-time",
        dest="no_equal_time",
        action="store_true",
        help="Hide the vertical marker at the equal-wall-clock baseline step",
    )
    p.add_argument(
        "--show_current",
        action="store_true",
        help="Also draw the (noisy) current-cost trace behind the best-so-far curve",
    )
    p.add_argument(
        "--out", type=str, default=None, help="Output basename (default: res/<FOLDER>/trace_<dim>)"
    )
    return p


def main() -> None:
    args = build_parser().parse_args()
    dtype = _DTYPE_MAP[args.dtype]

    steps_of = {
        "model": args.STEPS_MODEL or args.OUTER_STEPS,
        "baseline": args.STEPS_BASELINE or args.OUTER_STEPS,
    }

    set_seed(args.seed)

    model_path, ckpt_name = find_best_checkpoint_dir(args.FOLDER)
    print(f"Model: {model_path}\nCheckpoint: {ckpt_name}")

    cfg = {
        "PROBLEM_DIM": args.dim,
        "N_PROBLEMS": args.batch_size,
        "OUTER_STEPS": args.OUTER_STEPS,
        "TEST_OUTER_STEPS": args.OUTER_STEPS,
        "DEVICE": args.device,
        "SEED": args.seed,
        "LOAD_PB": True,
        "INIT": args.INIT,
        "MULTI_INIT": False,
        "DATA": args.DATA,
        "GREEDY": args.GREEDY,
        "METROPOLIS": args.METROPOLIS,
    }
    HP = init_problem_parameters(model_path, cfg)

    problem, init_x = initialize_test_problem(
        config=HP,
        test_dim=args.dim,
        n_test_problems=args.batch_size,
        init_method=args.INIT,
        data=args.DATA,
        device=args.device,
    )
    problem.set_heuristic(HP["HEURISTIC"])
    problem.set_feature_flags(HP["features"])
    input_dim = problem.get_input_dim()

    init_cost = torch.mean(problem.cost(init_x)).item()
    print(f"{args.batch_size} instances (dim {args.dim}), initial cost {init_cost:.4f}")

    actor = build_actor(HP, model_path, input_dim, device=args.device, seed=args.seed, dtype=dtype)
    warmup_cuda()

    # Both runs start from the identical problem state and initial solutions.
    orig_coords = problem.coords.clone()
    demands = problem.demands.clone()
    capacity = problem.capacity.clone()

    runs = {}
    for name, is_baseline in (("baseline", True), ("model", False)):
        n_steps = steps_of[name]
        problem.generate_params(orig_coords, demands, capacity)
        problem.init_parameters(init_x)
        set_seed(args.seed)  # same Metropolis random stream for both runs
        t0 = time.time()
        runs[name] = sa_trace(
            actor,
            problem,
            init_x,
            # Per-run budget: the Scheduler anneals over its own step_max, so each
            # run gets a complete INIT_TEMP -> STOP_TEMP schedule over its budget.
            {**HP, "TEST_OUTER_STEPS": n_steps},
            baseline=is_baseline,
            greedy=args.GREEDY,
            dtype=dtype,
            trace_every=args.trace_every,
            desc=f"{name} SA",
        )
        runs[name]["time"] = time.time() - t0
        print(
            f"{name:>8}: {n_steps} steps, final mean best cost "
            f"{runs[name]['best_cost'].mean().item():.4f}  "
            f"({runs[name]['time']:.1f}s)"
        )
        if args.device == "cuda":
            torch.cuda.empty_cache()

    # Long format: the two runs may have different budgets, so they cannot share
    # a single step column.
    df = pd.concat(
        [
            pd.DataFrame(
                {
                    "method": name,
                    "step": r["steps"].numpy(),
                    "step_frac": r["steps"].numpy() / steps_of[name],
                    "mean_best": r["mean_best"].numpy(),
                    "mean_current": r["mean_cur"].numpy(),
                }
            )
            for name, r in runs.items()
        ],
        ignore_index=True,
    )

    out = args.out or os.path.join("res", args.FOLDER, f"trace_{args.dim}")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_csv(f"{out}.csv", index=False)

    fin = {n: runs[n]["best_cost"].mean().item() for n in runs}

    # --- Equal-compute marker -------------------------------------------------
    # Baseline step reached after the same wall-clock time as the whole model run,
    # assuming a constant per-step cost: eq = T_model / (T_base / steps_base).
    # Deliberately coarse — it uses the two total runtimes, so it inherits any
    # fixed setup overhead in either run. Indicative, not a benchmark.
    eq_step = None
    if not args.no_equal_time and runs["baseline"]["time"] > 0:
        per_step_base = runs["baseline"]["time"] / steps_of["baseline"]
        eq_step = runs["model"]["time"] / per_step_base
        if eq_step > steps_of["baseline"]:
            print(
                f"Equal-time point is at baseline step ~{eq_step:.0f}, beyond its "
                f"{steps_of['baseline']} budget — marker omitted. "
                f"Raise --STEPS_BASELINE to show it."
            )
            eq_step = None
        else:
            g = df[df["method"] == "baseline"]
            cost_at_eq = float(np.interp(eq_step, g["step"], g["mean_best"]))
            print(
                f"Equal time ({runs['model']['time']:.1f}s): baseline step "
                f"~{eq_step:.0f}, mean best cost {cost_at_eq:.4f} "
                f"vs LG-SA {fin['model']:.4f}"
            )

    df_plot = df.copy()
    vline = eq_step
    if args.x_frac:
        df_plot["step"] = df_plot["step_frac"]
        if vline is not None:
            vline = vline / steps_of["baseline"]
    make_plot(
        df_plot,
        out,
        show_current=args.show_current,
        logx=args.logx,
        xlabel="Fraction of step budget" if args.x_frac else "SA step",
        vline=vline,
    )

    gap = 100.0 * (fin["baseline"] - fin["model"]) / fin["baseline"]
    print(
        f"\nFinal: LG-SA {fin['model']:.4f} ({steps_of['model']} steps) "
        f"vs SA {fin['baseline']:.4f} ({steps_of['baseline']} steps) "
        f"— {gap:+.2f}%"
    )
    print(f"Trace  -> {out}.csv")
    print(f"Figure -> {out}.pdf / {out}.png")


if __name__ == "__main__":
    main()
