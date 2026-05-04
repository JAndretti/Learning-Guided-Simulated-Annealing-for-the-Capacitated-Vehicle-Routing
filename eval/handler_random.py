import argparse
import os
import sys
import time

import pandas as pd
import torch
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from eval_io import find_models, get_HP_for_model, init_problem_parameters, save_results
from solver import build_actor, run_lgsa, set_seed, warmup_cuda

from init import initialize_test_problem


def add_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--dim", type=int, default=100, choices=[10, 20, 50, 100, 500, 1000]
    )
    parser.add_argument(
        "--DATA", type=str, default="nazari", choices=["nazari", "uchoa"]
    )
    parser.add_argument("--batch_size", type=int, default=10000)
    parser.add_argument(
        "--no-baseline", dest="BASELINE", action="store_false", default=True
    )
    parser.add_argument("--greedy", dest="GREEDY", action="store_true", default=False)
    parser.add_argument(
        "--no-metro", dest="METROPOLIS", action="store_false", default=True
    )


def _flatten(d: dict, parent: str = "", sep: str = ".") -> dict:
    items = []
    for k, v in d.items():
        key = f"{parent}{sep}{k}" if parent else k
        if isinstance(v, dict):
            items.extend(_flatten(v, key, sep).items())
        else:
            items.append((key, v))
    return dict(items)


def _differing_keys(model_names: list[str]) -> set[str]:
    """Return HP keys whose values differ across models."""
    all_hp = []
    for m in model_names:
        hp = get_HP_for_model(m)
        if hp:
            all_hp.append(_flatten(hp))
    diff = set()
    if all_hp:
        keys = set().union(*all_hp)
        for key in keys:
            values = set()
            for hp in all_hp:
                val = hp.get(key)
                if isinstance(val, list):
                    val = tuple(val)
                values.add(val)
            if len(values) > 1:
                diff.add(key)
    return diff


def _hp_row(model_name: str, diff_keys: set[str]) -> dict:
    HP = get_HP_for_model(model_name)
    if not HP:
        return {k: None for k in diff_keys}
    flat = _flatten(HP)
    return {k: flat.get(k) for k in diff_keys}


def run(args: argparse.Namespace) -> None:
    set_seed(args.seed)

    model_names = find_models(args.FOLDER)
    diff_keys = _differing_keys(model_names)

    cfg = {
        "PROBLEM_DIM": args.dim,
        "N_PROBLEMS": args.batch_size,
        "OUTER_STEPS": args.OUTER_STEPS,
        "DEVICE": args.device,
        "SEED": args.seed,
        "LOAD_PB": True,
        "INIT": args.INIT,
        "MULTI_INIT": False,
        "DATA": args.DATA,
        "BASELINE": args.BASELINE,
        "GREEDY": args.GREEDY,
        "METROPOLIS": args.METROPOLIS,
    }

    # Build shared problem + init_x once (data is shared across all models)
    problem, init_x = initialize_test_problem(
        config=cfg,
        test_dim=args.dim,
        n_test_problems=args.batch_size,
        init_method=args.INIT,
        data=args.DATA,
        device=args.device,
    )
    init_cost = torch.mean(problem.cost(init_x)).item()
    print(f"Problem initialised. Initial cost: {init_cost:.4f}")

    warmup_cuda()

    columns = [
        "model",
        "test_data",
        "initial_cost",
        "final_cost",
        "final_cost_baseline",
        "execution_time",
        "execution_time_baseline",
        "LGSA_steps",
        "SA_steps",
    ] + list(diff_keys)

    rows = []
    all_model_rows = []

    for model_name in tqdm(model_names, desc="Models", leave=False):
        HP = init_problem_parameters(model_name, cfg)

        # Update shared problem's features for this model
        problem.set_heuristic(HP["HEURISTIC"])
        problem.set_feature_flags(HP["features"])
        input_dim = problem.get_input_dim()

        actor = build_actor(
            HP, model_name, input_dim, device=args.device, seed=args.seed
        )

        # LGSA run
        t0 = time.time()
        test = run_lgsa(
            actor,
            problem,
            init_x,
            HP,
            outer_steps=args.OUTER_STEPS,
            baseline=False,
            greedy=args.GREEDY,
        )
        exec_time = time.time() - t0
        final_cost = torch.mean(problem.cost(test["best_x"].to(problem.device))).item()

        # Baseline run (pure SA)
        if args.BASELINE:
            t0 = time.time()
            test_bl = run_lgsa(
                actor,
                problem,
                init_x,
                HP,
                outer_steps=args.OUTER_STEPS,
                baseline=True,
                greedy=False,
            )
            exec_time_bl = time.time() - t0
            final_cost_bl = torch.mean(
                problem.cost(test_bl["best_x"].to(problem.device))
            ).item()
        else:
            final_cost_bl = float("nan")
            exec_time_bl = float("nan")

        if args.device == "cuda":
            torch.cuda.empty_cache()

        row = {
            "model": os.path.basename(model_name),
            "test_data": args.DATA,
            "initial_cost": init_cost,
            "final_cost": final_cost,
            "final_cost_baseline": final_cost_bl,
            "execution_time": exec_time,
            "execution_time_baseline": exec_time_bl,
            "LGSA_steps": args.OUTER_STEPS,
            "SA_steps": args.OUTER_STEPS if args.BASELINE else float("nan"),
            **_hp_row(model_name, diff_keys),
        }
        rows.append(row)
        all_model_rows.append(
            {
                "model": f"{args.FOLDER}/{os.path.basename(model_name)}",
                "final_cost": final_cost,
            }
        )

    df = pd.DataFrame(rows, columns=columns).drop_duplicates(
        subset=["model"], keep="first"
    )
    df_all = pd.DataFrame(all_model_rows).drop_duplicates(
        subset=["model"], keep="first"
    )

    base_path = f"res/{args.FOLDER}"
    out = save_results(df, base_path, f"res_model_{args.dim}")
    out_all = save_results(df_all, base_path, f"res_all_model_{args.dim}")
    print(f"Results saved to {out}")
    print(f"All-model summary saved to {out_all}")
    print(df.head())
