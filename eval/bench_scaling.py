"""
Scaling sweep: final cost and runtime as a function of step budget and problem
dimension.  (Old paper: Table "Scaling Analysis" + Figure "Scaling Analysis (a)/(b)".)

Each (dim, steps) cell is an independent run: the temperature schedule anneals
INIT_TEMP -> STOP_TEMP over whatever budget it is given, so a 1,000-step run is
*not* a prefix of a 10,000-step run and the cells cannot be read off a single
long trace.

Usage
-----
    uv run eval/bench_scaling.py --FOLDER BEST \
        --dims 10,20,50,100 --steps 100,1000,5000,10000 --batch_size 10000

    # heavy tail of the sweep, resumable
    uv run eval/bench_scaling.py --FOLDER BEST --dims 50,100 \
        --steps 100000,200000,500000,1000000 --batch_size 1000 --resume

Output: res/<FOLDER>/bench/scaling.csv  (one row per dim x steps cell)
"""

import argparse

import torch
from bench_common import (
    DTYPE_MAP,
    add_common_args,
    append_row,
    batch_for,
    batch_spec,
    bench_path,
    build_problem_and_actor,
    int_list,
    load_done,
    out_name,
    prepare_run,
    timed_lgsa,
)
from tqdm import tqdm

KEYS = ["dim", "steps", "batch_size", "dtype"]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_common_args(p)
    p.add_argument("--dims", type=int_list, default="10,20,50,100")
    p.add_argument("--steps", type=int_list, default="100,1000,5000,10000")
    p.add_argument(
        "--batch_size",
        type=batch_spec,
        default="10000",
        help="Instances per cell. '10000' or '10000,500=1000,1000=200' for per-dim overrides",
    )
    p.add_argument(
        "--baseline",
        action="store_true",
        help="Also measure pure SA (no actor) on the same cells",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()
    prepare_run(args)
    dtype = DTYPE_MAP[args.dtype]

    path = bench_path(args.FOLDER, out_name("scaling", args.tag))
    done = load_done(path, KEYS) if args.resume else set()

    for dim in args.dims:
        batch = batch_for(args.batch_size, dim)
        todo = [s for s in args.steps if (dim, s, batch, args.dtype) not in done]
        if not todo:
            print(f"dim={dim}: all cells present, skipping")
            continue

        # One problem/actor build per dimension, reused across step budgets: the
        # instances are identical down the column, only the budget changes.
        actor, problem, init_x, HP, setup_time = build_problem_and_actor(
            args.FOLDER,
            dim,
            batch,
            max(todo),
            args.device,
            args.seed,
            dtype=dtype,
            init=args.INIT,
            data=args.DATA,
        )
        init_cost = torch.mean(problem.cost(init_x)).item()
        print(f"dim={dim}  batch={batch}  init_cost={init_cost:.4f}  setup={setup_time:.1f}s")

        for steps in tqdm(todo, desc=f"N={dim}", leave=False):
            cost, sa_time = timed_lgsa(actor, problem, init_x, HP, steps, args.device, dtype=dtype)
            row = {
                "dim": dim,
                "steps": steps,
                "batch_size": batch,
                "dtype": args.dtype,
                "init_cost": init_cost,
                "cost": cost,
                "sa_time": sa_time,
                "time_per_instance_ms": 1000.0 * sa_time / batch,
                "setup_time": setup_time,
                "device": args.device,
                "init": args.INIT,
                "data": args.DATA,
            }
            if args.baseline:
                cost_bl, time_bl = timed_lgsa(
                    actor, problem, init_x, HP, steps, args.device, dtype=dtype, baseline=True
                )
                row["cost_baseline"] = cost_bl
                row["sa_time_baseline"] = time_bl
            append_row(path, row)
            tqdm.write(
                f"  N={dim:>6} steps={steps:>9,} cost={cost:.4f} time={sa_time:8.2f}s"
                + (f" | SA {row['cost_baseline']:.4f}" if args.baseline else "")
            )

        del actor, problem, init_x
        if args.device == "cuda":
            torch.cuda.empty_cache()

    print(f"\nResults -> {path}")


if __name__ == "__main__":
    main()
