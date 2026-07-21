"""
Inference-precision study: float32 vs. float16 vs. bfloat16.
(Old paper: Table "Effect of inference-time numerical precision on LG-SA".)

All precisions solve the *same* instances from the *same* initial solutions, and
the RNG is reseeded before each run so the Metropolis random stream matches;
what differs is only the arithmetic. The actor weights are cast via
`actor.to(dtype)` and the SA loop casts its float tensors — integer route/action
tensors stay int64 (handled inside sa_test).

float32 is always measured first and used as the reference for the delta-cost
and speedup columns.

Usage
-----
    uv run eval/bench_precision.py --FOLDER BEST --dim 100 \
        --batch_size 10000 --steps 10000

Output: res/<FOLDER>/bench/precision.csv  (one row per dtype)
"""

import argparse
import os

import pandas as pd
import torch
from bench_common import (
    DTYPE_MAP,
    add_common_args,
    append_row,
    bench_path,
    build_problem_and_actor,
    load_done,
    out_name,
    prepare_run,
    timed_lgsa,
)
from solver import set_seed

KEYS = ["dim", "steps", "batch_size", "precision"]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_common_args(p)
    p.add_argument("--dim", type=int, default=100)
    p.add_argument("--steps", type=int, default=10000)
    p.add_argument("--batch_size", type=int, default=10000)
    p.add_argument(
        "--precisions",
        type=lambda s: [x for x in s.replace(" ", "").split(",") if x],
        default="float32,float16,bfloat16",
        help="Comma-separated subset of float32,float16,bfloat16 (float32 first = reference)",
    )
    p.add_argument(
        "--repeats", type=int, default=1, help="Timed repetitions per precision (median reported)"
    )
    return p


def main() -> None:
    args = build_parser().parse_args()
    prepare_run(args)

    path = bench_path(args.FOLDER, out_name("precision", args.tag))
    done = load_done(path, KEYS) if args.resume else set()

    # On a resumed run the float32 row may already be on disk; reuse it as the
    # reference so the delta/speedup columns stay comparable across sessions.
    ref_cost = ref_time = None
    if os.path.exists(path):
        prev = pd.read_csv(path)
        prev = prev[
            (prev.dim == args.dim)
            & (prev.steps == args.steps)
            & (prev.batch_size == args.batch_size)
            & (prev.precision == "float32")
        ]
        if not prev.empty:
            ref_cost = float(prev.cost.iloc[-1])
            ref_time = float(prev.sa_time.iloc[-1])

    for precision in args.precisions:
        if precision not in DTYPE_MAP:
            raise ValueError(f"Unknown precision {precision!r}")
        if (args.dim, args.steps, args.batch_size, precision) in done:
            print(f"{precision}: already measured, skipping")
            continue

        dtype = DTYPE_MAP[precision]
        # Rebuild per precision: the actor is cast in build_actor, and re-seeding
        # here gives every precision the same instances and the same RNG stream.
        set_seed(args.seed)
        actor, problem, init_x, HP, setup_time = build_problem_and_actor(
            args.FOLDER,
            args.dim,
            args.batch_size,
            args.steps,
            args.device,
            args.seed,
            dtype=dtype,
            init=args.INIT,
            data=args.DATA,
        )

        times, cost = [], float("nan")
        for _ in range(args.repeats):
            set_seed(args.seed)
            cost, sa_time = timed_lgsa(
                actor, problem, init_x, HP, args.steps, args.device, dtype=dtype
            )
            times.append(sa_time)
        sa_time = sorted(times)[len(times) // 2]

        if precision == "float32" or ref_cost is None:
            ref_cost, ref_time = cost, sa_time

        append_row(
            path,
            {
                "dim": args.dim,
                "steps": args.steps,
                "batch_size": args.batch_size,
                "precision": precision,
                "cost": cost,
                "delta_cost": cost - ref_cost,
                "sa_time": sa_time,
                "speedup": ref_time / sa_time,
                "setup_time": setup_time,
                "device": args.device,
                "repeats": args.repeats,
            },
        )
        print(
            f"{precision:>9}: cost={cost:.4f} ({cost - ref_cost:+.4f})  "
            f"time={sa_time:.1f}s  speedup={ref_time / sa_time:.2f}x"
        )

        del actor, problem, init_x
        if args.device == "cuda":
            torch.cuda.empty_cache()

    print(f"\nResults -> {path}")


if __name__ == "__main__":
    main()
