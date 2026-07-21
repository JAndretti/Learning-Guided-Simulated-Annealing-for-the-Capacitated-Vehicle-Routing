"""
Shared plumbing for the scaling / timing benchmarks (bench_*.py).

Every benchmark script here follows the same contract:

  * it appends one row per measured configuration to a CSV under
    ``res/<FOLDER>/bench/<name>.csv`` (created on first write),
  * re-running with ``--resume`` skips configurations already present in that
    CSV, so a long sweep (1M steps, N=10,000) can be interrupted and continued,
  * plotting and LaTeX-table emission live in ``bench_plots.py`` and read only
    those CSVs — no re-running needed to redo a figure.

Timing convention: only the SA loop is timed (``sa_time``). Problem setup
(distance matrix, KNN, initial solution) is timed separately as ``setup_time``
and never folded into the reported numbers, since the paper's timings describe
the search itself. CUDA is synchronised on both sides of every timed region.
"""

import os
import sys
import time
from typing import Any

import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from eval_io import init_problem_parameters
from solver import build_actor, run_lgsa, set_seed, warmup_cuda
from trace_curve import find_best_checkpoint_dir

from init import initialize_test_problem
from problem import CVRP

DTYPE_MAP = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}

# Nazari convention (see generated_nazari_problem/generate_nazari_pb.py):
# demands ~ U{1..9}, capacity fixed per dimension, coords ~ U[0,1]^2.
NAZARI_CAPACITY = {10: 20, 20: 30, 50: 40, 100: 50, 500: 50, 1000: 50}
DEFAULT_CAPACITY = 50

# Dimensions with a pre-generated test set on disk; anything else is sampled.
PREGENERATED_DIMS = (10, 20, 50, 100, 500, 1000)


# ============================================================================
# TIMING
# ============================================================================


def sync(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


class Timer:
    """Context manager giving a CUDA-synchronised wall-clock duration in seconds."""

    def __init__(self, device: str) -> None:
        self.device = device
        self.elapsed = float("nan")

    def __enter__(self) -> "Timer":
        sync(self.device)
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc: object) -> None:
        sync(self.device)
        self.elapsed = time.perf_counter() - self._t0


# ============================================================================
# MODEL / PROBLEM CONSTRUCTION
# ============================================================================


def base_cfg(
    dim: int,
    n_problems: int,
    steps: int,
    device: str,
    seed: int,
    init: str = "random",
    data: str = "nazari",
) -> dict:
    return {
        "PROBLEM_DIM": dim,
        "N_PROBLEMS": n_problems,
        "OUTER_STEPS": steps,
        "TEST_OUTER_STEPS": steps,
        "DEVICE": device,
        "SEED": seed,
        "LOAD_PB": True,
        "INIT": init,
        "MULTI_INIT": False,
        "DATA": data,
        "BASELINE": False,
        "GREEDY": False,
        "METROPOLIS": True,
    }


def load_hp(folder: str, cfg: dict) -> tuple[dict, str, str]:
    """Return (HP merged with cfg, model_dir, checkpoint_name) for the best checkpoint."""
    model_dir, ckpt = find_best_checkpoint_dir(folder)
    return init_problem_parameters(model_dir, cfg), model_dir, ckpt


def make_random_problem(
    HP: dict,
    dim: int,
    n_problems: int,
    device: str,
    seed: int,
    init: str = "random",
) -> tuple[CVRP, torch.Tensor]:
    """
    Sample fresh Nazari-style instances of *any* dimension.

    ``initialize_test_problem`` can only serve the pre-generated dimensions; the
    extreme-scale timing sweep needs N far beyond those, so instances are drawn
    here with the same distribution the training/test sets use.

    Memory note: ``CVRP.set_params`` materialises a dense [B, N+1, N+1] distance
    matrix plus an argsort of it. At N=10,000 that is ~0.4 GB per instance for
    the matrix alone and several GB transiently for the ranks — keep B small.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    coords = torch.rand(n_problems, dim + 1, 2, generator=g)
    demands = torch.randint(1, 10, (n_problems, dim + 1), generator=g)
    demands[:, 0] = 0
    capacity = NAZARI_CAPACITY.get(dim, DEFAULT_CAPACITY)
    capacities = torch.full((n_problems, 1), capacity)

    problem = CVRP(dim=dim, n_problems=n_problems, device=device, params=HP)
    problem.manual_seed(seed)
    problem.set_heuristic(HP["HEURISTIC"])
    problem.set_feature_flags(HP["features"])
    problem.generate_params(coords.to(device), demands.to(device), capacities.to(device))
    init_x = problem.generate_init_state(init, False)
    return problem, init_x


def build_problem_and_actor(
    folder: str,
    dim: int,
    n_problems: int,
    steps: int,
    device: str,
    seed: int,
    *,
    dtype: torch.dtype = torch.float32,
    init: str = "random",
    data: str = "nazari",
    source: str = "default",
    synthetic: bool = False,
) -> tuple[Any, CVRP, torch.Tensor, dict, float]:
    """
    Build (actor, problem, init_x, HP, setup_time).

    ``synthetic=True`` samples instances instead of loading the test set — the
    only option for dimensions without a pre-generated file.
    """
    cfg = base_cfg(dim, n_problems, steps, device, seed, init=init, data=data)
    HP, model_dir, ckpt = load_hp(folder, cfg)

    use_synthetic = synthetic or dim not in PREGENERATED_DIMS
    with Timer(device) as t:
        if use_synthetic:
            problem, init_x = make_random_problem(HP, dim, n_problems, device, seed, init)
        else:
            problem, init_x = initialize_test_problem(
                config=HP,
                test_dim=dim,
                n_test_problems=n_problems,
                init_method=init,
                data=data,
                device=device,
                source=source,
            )
            problem.set_heuristic(HP["HEURISTIC"])
            problem.set_feature_flags(HP["features"])

    actor = build_actor(
        HP, model_dir, problem.get_input_dim(), device=device, seed=seed, dtype=dtype
    )
    print(f"model={model_dir}\n  checkpoint={ckpt}  synthetic_data={use_synthetic}")
    return actor, problem, init_x, HP, t.elapsed


def timed_lgsa(
    actor: Any,
    problem: CVRP,
    init_x: torch.Tensor,
    HP: dict,
    steps: int,
    device: str,
    *,
    dtype: torch.dtype = torch.float32,
    baseline: bool = False,
    greedy: bool = False,
) -> tuple[float, float]:
    """Run LG-SA once and return (mean final cost, sa_time seconds)."""
    with Timer(device) as t:
        res = run_lgsa(
            actor,
            problem,
            init_x,
            HP,
            outer_steps=steps,
            baseline=baseline,
            greedy=greedy,
            dtype=dtype,
        )
    cost = torch.mean(problem.cost(res["best_x"].to(problem.device))).item()
    return cost, t.elapsed


# ============================================================================
# INCREMENTAL CSV
# ============================================================================


def bench_path(folder: str, name: str) -> str:
    path = os.path.join("res", folder, "bench", f"{name}.csv")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def load_done(path: str, keys: list[str]) -> set[tuple]:
    """Return the set of key-tuples already measured in an existing CSV."""
    if not os.path.exists(path):
        return set()
    df = pd.read_csv(path)
    if df.empty or not set(keys).issubset(df.columns):
        return set()
    return {tuple(row) for row in df[keys].itertuples(index=False, name=None)}


def append_row(path: str, row: dict) -> None:
    """Append one row, writing the header only when the file is created."""
    header = not os.path.exists(path)
    pd.DataFrame([row]).to_csv(path, mode="a", header=header, index=False)


# ============================================================================
# CLI HELPERS
# ============================================================================


def int_list(spec: str) -> list[int]:
    """'100,1000,10000' -> [100, 1000, 10000]"""
    return [int(x) for x in spec.replace(" ", "").split(",") if x]


def batch_spec(spec: str) -> dict[int | None, int]:
    """
    Parse a batch-size spec into {dim: batch, None: default}.

    '10000'                 -> every dimension uses 10,000 instances
    '1000,500=200,1000=50'  -> default 1,000; N=500 uses 200; N=1,000 uses 50
    """
    out: dict[int | None, int] = {}
    for part in spec.replace(" ", "").split(","):
        if not part:
            continue
        if "=" in part:
            k, v = part.split("=")
            out[int(k)] = int(v)
        else:
            out[None] = int(part)
    if None not in out:
        out[None] = 10000
    return out


def batch_for(spec: dict[int | None, int], dim: int) -> int:
    return spec.get(dim, spec[None])


def add_common_args(parser: Any) -> None:
    parser.add_argument("--FOLDER", type=str, default="BEST", help="Model folder under wandb/LGSA/")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--INIT",
        type=str,
        default="random",
        choices=["random", "isolate", "sweep", "nearest_neighbor", "Clark_and_Wright"],
    )
    parser.add_argument("--DATA", type=str, default="nazari", choices=["nazari", "uchoa"])
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--dtype", type=str, default="float32", choices=list(DTYPE_MAP))
    parser.add_argument(
        "--tag", type=str, default="", help="Suffix for the output CSV name (e.g. a GPU name)"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip configurations already present in the output CSV",
    )


def prepare_run(args: Any) -> None:
    set_seed(args.seed)
    warmup_cuda()


def out_name(base: str, tag: str) -> str:
    return f"{base}_{tag}" if tag else base
