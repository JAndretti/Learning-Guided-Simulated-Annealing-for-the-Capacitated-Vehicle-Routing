"""A-bis factorial: initialisation x proposal x budget x seed.

Drives the cross product defined in
docs/specs/2026-07-30-abis-stronger-baseline-design.md, writing one CSV row and
one per-instance cost vector per run. The per-instance vectors are what make the
paired bootstrap in eval/abis_analysis.py possible; means alone cannot support it.

Usage — reported factorial (see docs/plans/2026-07-30-abis-factorial.md for the
tuning invocations):

    uv run eval/run.py --dataset factorial \
        --n_instances 1000 --offset 0 \
        --inits random,Clark_and_Wright \
        --proposals blind,score1,score2,lgsa \
        --budgets 1000,2000,5000,10000,20000 \
        --seeds 1234,1235,1236 \
        --alpha1 <tuned> --alpha2 <tuned> --resume
"""

import argparse
import itertools
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from eval_io import find_model, get_HP_for_model
from solver import build_actor, run_lgsa, set_seed, warmup_cuda

from init import initialize_test_problem
from model import ScoreActor

# Every flag present and False except `meta`: build_state_components defaults
# static/topology/meta to True when a key is absent, so an empty dict would NOT
# give a bare state. `meta` stays on because ScoreActor reads the temperature
# from it. Result: state = [node index, temp_norm, progress] -> [B, L, 3].
SCORE_FEATURE_FLAGS = {
    "static": False,
    "topology": False,
    "neighbor_rank": False,
    "demand_max": False,
    "density5": False,
    "density10%": False,
    "density33%": False,
    "detour": False,
    "centroid": False,
    "route_pct": False,
    "slack": False,
    "node_pct": False,
    "meta": True,
}

PROPOSALS = ("blind", "score1", "score2", "lgsa")

# Deterministic constructions are built once and reused across seeds; stochastic
# ones are rebuilt per seed so that init variance is part of the seed's variance.
DETERMINISTIC_INITS = {"Clark_and_Wright", "Clark_and_Wright_reversal", "sweep", "isolate"}

CSV_COLUMNS = [
    "init", "proposal", "budget", "seed",
    "alpha1", "alpha2", "init_temp", "stop_temp",
    "init_cost", "final_cost",
    "setup_time", "sa_time", "total_time",
    "n_instances", "offset", "dim", "dtype", "device", "cpu_cores",
    "checkpoint", "costs_file",
]
# Identifies a run for --resume. Excludes timings and results.
# stop_temp belongs here: a schedule cooled to a different floor is a different
# run, and omitting it would let --resume silently skip it as already done.
KEY_COLUMNS = [
    "init", "proposal", "budget", "seed",
    "alpha1", "alpha2", "init_temp", "stop_temp", "n_instances", "offset", "dim",
]


def add_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dim", type=int, default=100)
    parser.add_argument("--n_instances", type=int, default=1000)
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="First instance index. Reported set: 0. Tuning set: 1000.",
    )
    parser.add_argument("--inits", type=str, default="random,Clark_and_Wright")
    parser.add_argument("--proposals", type=str, default="blind,score1,score2,lgsa")
    parser.add_argument("--budgets", type=str, default="1000,2000,5000,10000,20000")
    parser.add_argument("--seeds", type=str, default="1234,1235,1236")
    parser.add_argument(
        "--alpha1", type=str, default="0.1",
        help="Comma-separated stage-1 multipliers; a list sweeps them.",
    )
    parser.add_argument(
        "--alpha2", type=str, default="0.1",
        help="Comma-separated stage-2 multipliers; a list sweeps them.",
    )
    parser.add_argument(
        "--init_temp", type=str, default="1.0",
        help="Comma-separated INIT_TEMP values; a list sweeps them. Applies to "
             "the non-learned proposals only — 'lgsa' always uses its "
             "checkpoint's INIT_TEMP, since it was trained under that range.",
    )
    parser.add_argument(
        "--stop_temp", type=str, default="",
        help="Comma-separated STOP_TEMP values; a list sweeps them. Empty (the "
             "default) uses the checkpoint's STOP_TEMP. Applies to the "
             "non-learned proposals only, for the same reason as --init_temp. "
             "Needed because INIT_TEMP cannot usefully go below STOP_TEMP: "
             "testing a genuinely cold schedule means lowering both.",
    )
    parser.add_argument("--out_dir", type=str, default="res/abis")
    parser.add_argument("--csv_name", type=str, default="factorial")
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip configurations already present in the CSV.",
    )


def _floats(s: str) -> list[float]:
    return [float(v) for v in s.split(",") if v != ""]


def _ints(s: str) -> list[int]:
    return [int(v) for v in s.split(",") if v != ""]


def _strs(s: str) -> list[str]:
    return [v for v in s.split(",") if v != ""]


def _fmt(v: float) -> str:
    """Filename-safe number: 0.1 -> 0p1, 1.0 -> 1, 20000 -> 20000, nan -> na."""
    if isinstance(v, float) and np.isnan(v):
        return "na"
    return ("%g" % v).replace(".", "p").replace("-", "m")


def _key(record: dict | tuple) -> tuple:
    """Resume key with NaN normalised to a string.

    Unused alphas are NaN in the CSV (spec 6.4), and NaN != NaN, so a raw tuple
    key would never match itself — every resume would re-run all the `blind`
    rows. Mapping NaN to a sentinel makes the comparison reflexive while leaving
    the CSV as the spec requires.
    """
    values = record if isinstance(record, tuple) else tuple(record[k] for k in KEY_COLUMNS)
    return tuple(
        "nan" if isinstance(v, float) and np.isnan(v) else v for v in values
    )


def _costs_filename(cell: dict) -> str:
    return (
        f"{cell['init']}_{cell['proposal']}"
        f"_b{cell['budget']}_s{cell['seed']}"
        f"_a1-{_fmt(cell['alpha1'])}_a2-{_fmt(cell['alpha2'])}"
        f"_T{_fmt(cell['init_temp'])}-{_fmt(cell['stop_temp'])}"
        f"_n{cell['n_instances']}_o{cell['offset']}.npy"
    )


def _cell_alphas(proposal: str, a1: float, a2: float) -> tuple[float, float]:
    """Alphas that actually affect this proposal; unused ones become NaN.

    Keeps the resume key from treating 'blind at alpha1=0.1' and
    'blind at alpha1=1.0' as two different runs — they are the same run.
    """
    if proposal == "score1":
        return a1, float("nan")
    if proposal == "score2":
        return a1, a2
    return float("nan"), float("nan")  # blind, lgsa


def _cell_init_temp(proposal: str, sweep_temp: float, ckpt_temp: float) -> float:
    """'lgsa' keeps its checkpoint's temperature range; others take the swept one."""
    return ckpt_temp if proposal == "lgsa" else sweep_temp


def _cell_stop_temp(proposal: str, sweep_temp: float, ckpt_temp: float) -> float:
    """Same rule as _cell_init_temp, for the cooling floor."""
    return ckpt_temp if proposal == "lgsa" else sweep_temp


def run(args: argparse.Namespace) -> None:
    inits = _strs(args.inits)
    proposals = _strs(args.proposals)
    budgets = _ints(args.budgets)
    seeds = _ints(args.seeds)
    alpha1s = _floats(args.alpha1)
    alpha2s = _floats(args.alpha2)
    init_temps = _floats(args.init_temp)

    unknown = set(proposals) - set(PROPOSALS)
    if unknown:
        raise ValueError(f"Unknown proposals {sorted(unknown)}; choose from {PROPOSALS}")

    model_dir = find_model(args.FOLDER)
    HP = get_HP_for_model(model_dir)
    ckpt_init_temp = float(HP["INIT_TEMP"])
    ckpt_stop_temp = float(HP["STOP_TEMP"])
    stop_temps = _floats(args.stop_temp) if args.stop_temp else [ckpt_stop_temp]

    bad = [(i, s) for i in init_temps for s in stop_temps if i <= s]
    if bad:
        raise ValueError(
            f"INIT_TEMP must exceed STOP_TEMP (the schedule cools downward); "
            f"offending (init, stop) pairs: {bad}"
        )

    costs_dir = os.path.join(args.out_dir, "costs")
    os.makedirs(costs_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, f"{args.csv_name}.csv")

    done: set[tuple] = set()
    if args.resume and os.path.exists(csv_path):
        prev = pd.read_csv(csv_path)
        done = {
            _key(r) for r in prev[KEY_COLUMNS].itertuples(index=False, name=None)
        }
        print(f"Resume: {len(done)} runs already in {csv_path}")

    # --- enumerate cells, dropping duplicates created by irrelevant sweeps ---
    cells: list[dict] = []
    seen: set[tuple] = set()
    for init, proposal, budget, seed, a1, a2, t0, t1 in itertools.product(
        inits, proposals, budgets, seeds, alpha1s, alpha2s, init_temps, stop_temps
    ):
        ca1, ca2 = _cell_alphas(proposal, a1, a2)
        cell = {
            "init": init,
            "proposal": proposal,
            "budget": budget,
            "seed": seed,
            "alpha1": ca1,
            "alpha2": ca2,
            "init_temp": _cell_init_temp(proposal, t0, ckpt_init_temp),
            "stop_temp": _cell_stop_temp(proposal, t1, ckpt_stop_temp),
            "n_instances": args.n_instances,
            "offset": args.offset,
            "dim": args.dim,
        }
        key = _key(cell)
        if key in seen:
            continue
        seen.add(key)
        cells.append(cell)

    todo = [c for c in cells if _key(c) not in done]
    print(f"{len(cells)} configurations, {len(todo)} to run")
    if not todo:
        return

    # --- shared problem, built once at the requested slice ---
    cfg = {
        "PROBLEM_DIM": args.dim,
        "N_PROBLEMS": args.n_instances,
        "DEVICE": args.device,
        "SEED": args.seed,
        "HEURISTIC": HP["HEURISTIC"],
        "features": HP["features"],
    }
    set_seed(args.seed)
    problem, _ = initialize_test_problem(
        config=cfg,
        test_dim=args.dim,
        n_test_problems=args.n_instances,
        init_method="random",
        data="nazari",
        device=args.device,
        source="default",
        offset=args.offset,
    )
    problem.set_heuristic(HP["HEURISTIC"])
    warmup_cuda()

    lgsa_input_dim = None
    if "lgsa" in proposals:
        problem.set_feature_flags(HP["features"])
        lgsa_input_dim = problem.get_input_dim()

    init_cache: dict[tuple, tuple[torch.Tensor, float, float]] = {}

    def get_init(init: str, seed: int) -> tuple[torch.Tensor, float, float]:
        """(init_x, setup_time, init_cost), cached.

        Deterministic constructions are keyed on the init alone, so Clarke-Wright
        is paid once. Its measured time is still reported in full on every row
        that uses it: the question a row answers is what it costs to solve this
        instance set with this method, and that includes construction.
        """
        key = (init,) if init in DETERMINISTIC_INITS else (init, seed)
        if key not in init_cache:
            set_seed(seed)
            problem.manual_seed(seed)
            t0 = time.time()
            init_x = problem.generate_init_state(init, False)
            if args.device == "cuda":
                torch.cuda.synchronize()
            setup_time = time.time() - t0
            init_cost = problem.cost(init_x).mean().item()
            init_cache[key] = (init_x, setup_time, init_cost)
            print(f"  built {init} (seed {seed}): cost {init_cost:.4f} in {setup_time:.2f}s")
        return init_cache[key]

    rows = []
    cpu_cores = os.cpu_count()

    for cell in tqdm(todo, desc="cells"):
        proposal = cell["proposal"]
        init_x, setup_time, init_cost = get_init(cell["init"], cell["seed"])

        run_hp = {
            **HP,
            "INIT_TEMP": cell["init_temp"],
            "STOP_TEMP": cell["stop_temp"],
            "METROPOLIS": True,
            "GREEDY": False,
        }

        set_seed(cell["seed"])
        if proposal == "lgsa":
            problem.set_feature_flags(HP["features"])
            actor = build_actor(
                run_hp, model_dir, lgsa_input_dim,
                device=args.device, seed=cell["seed"], dtype=args.torch_dtype,
            )
            baseline = False
        else:
            problem.set_feature_flags(SCORE_FEATURE_FLAGS)
            if proposal == "blind":
                # Uniform x uniform via the inherited, already-validated
                # baseline_sample path; baseline=True also skips state building.
                actor = ScoreActor(
                    "score1", 1.0, 1.0, cell["init_temp"], cell["stop_temp"],
                    device=args.device, method=HP["UPDATE_METHOD"],
                )
                baseline = True
            else:
                actor = ScoreActor(
                    proposal, cell["alpha1"],
                    1.0 if proposal == "score1" else cell["alpha2"],
                    cell["init_temp"], cell["stop_temp"],
                    device=args.device, method=HP["UPDATE_METHOD"],
                )
                baseline = False
        actor.manual_seed(cell["seed"])

        t0 = time.time()
        result = run_lgsa(
            actor, problem, init_x, run_hp,
            outer_steps=cell["budget"],
            baseline=baseline,
            greedy=False,
            dtype=args.torch_dtype,
        )
        if args.device == "cuda":
            torch.cuda.synchronize()
        sa_time = time.time() - t0

        per_instance = problem.cost(result["best_x"].to(problem.device))  # [n_instances]
        costs_file = _costs_filename(cell)
        np.save(os.path.join(costs_dir, costs_file), per_instance.cpu().numpy())

        rows.append({
            **cell,
            "init_cost": init_cost,
            "final_cost": per_instance.mean().item(),
            "setup_time": setup_time,
            "sa_time": sa_time,
            "total_time": setup_time + sa_time,
            "dtype": args.dtype,
            "device": args.device,
            "cpu_cores": cpu_cores,
            "checkpoint": os.path.basename(model_dir) if proposal == "lgsa" else "",
            "costs_file": costs_file,
        })

        # Append after every run so an interrupted sweep loses at most one cell.
        df = pd.DataFrame(rows, columns=CSV_COLUMNS)
        header = not os.path.exists(csv_path)
        df.tail(1).to_csv(csv_path, mode="a", header=header, index=False)

        if args.device == "cuda":
            torch.cuda.empty_cache()

    print(f"Wrote {len(rows)} rows to {csv_path}")
    print(pd.DataFrame(rows, columns=CSV_COLUMNS)[
        ["init", "proposal", "budget", "seed", "final_cost", "total_time"]
    ].to_string(index=False))
