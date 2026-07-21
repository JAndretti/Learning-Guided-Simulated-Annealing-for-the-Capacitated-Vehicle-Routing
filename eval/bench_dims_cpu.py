"""
Extreme-scale dimension sweep: sequential per-instance runtime vs. N.
(Old paper: Figure "Sequential CPU execution time as a function of problem dimension".)

Instances are processed one at a time (batch = 1) so the measurement reflects
the cost of the search itself rather than GPU batching throughput; the default
device is therefore CPU. Dimensions beyond the pre-generated test sets are
sampled from the Nazari distribution on the fly.

Memory: CVRP builds a dense [1, N+1, N+1] distance matrix and argsorts it.
N=10,000 needs ~0.4 GB for the matrix and a few GB transiently for the ranks —
that setup cost is timed separately (`setup_time`) and excluded from the
reported search time.

Usage
-----
    uv run eval/bench_dims_cpu.py --FOLDER BEST \
        --dims 100,200,500,1000,2000,5000,10000 --steps 1000,10000 --repeats 3

Output: res/<FOLDER>/bench/dims_cpu.csv  (one row per dim x steps x repeat)
"""

import argparse

import torch
from bench_common import (
    DTYPE_MAP,
    add_common_args,
    append_row,
    bench_path,
    build_problem_and_actor,
    int_list,
    load_done,
    out_name,
    prepare_run,
    timed_lgsa,
)
from tqdm import tqdm

KEYS = ["dim", "steps", "repeat", "device"]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_common_args(p)
    p.set_defaults(device="cpu")
    p.add_argument("--dims", type=int_list, default="100,200,500,1000,2000,5000,10000")
    p.add_argument("--steps", type=int_list, default="1000,10000")
    p.add_argument(
        "--repeats", type=int, default=3, help="Independent instances per (dim, steps) cell"
    )
    p.add_argument(
        "--threads", type=int, default=0, help="torch CPU threads (0 = leave torch's default)"
    )
    return p


def main() -> None:
    args = build_parser().parse_args()
    if args.threads > 0:
        torch.set_num_threads(args.threads)
    prepare_run(args)
    dtype = DTYPE_MAP[args.dtype]

    path = bench_path(args.FOLDER, out_name("dims_cpu", args.tag))
    done = load_done(path, KEYS) if args.resume else set()

    for dim in args.dims:
        for repeat in range(args.repeats):
            todo = [s for s in args.steps if (dim, s, repeat, args.device) not in done]
            if not todo:
                continue

            # A fresh instance per repeat: seed varies, so the sweep averages over
            # instances rather than re-timing the same one.
            actor, problem, init_x, HP, setup_time = build_problem_and_actor(
                args.FOLDER,
                dim,
                1,
                max(todo),
                args.device,
                args.seed + repeat,
                dtype=dtype,
                init=args.INIT,
                data=args.DATA,
                synthetic=True,
            )
            init_cost = problem.cost(init_x).mean().item()

            for steps in tqdm(todo, desc=f"N={dim} r{repeat}", leave=False):
                cost, sa_time = timed_lgsa(
                    actor, problem, init_x, HP, steps, args.device, dtype=dtype
                )
                append_row(
                    path,
                    {
                        "dim": dim,
                        "steps": steps,
                        "repeat": repeat,
                        "device": args.device,
                        "threads": torch.get_num_threads(),
                        "init_cost": init_cost,
                        "cost": cost,
                        "sa_time": sa_time,
                        "ms_per_step": 1000.0 * sa_time / steps,
                        "setup_time": setup_time,
                        "dtype": args.dtype,
                    },
                )
                tqdm.write(
                    f"  N={dim:>6} steps={steps:>7,} r={repeat} "
                    f"time={sa_time:8.2f}s (setup {setup_time:6.2f}s) cost={cost:.3f}"
                )

            del actor, problem, init_x
            if args.device == "cuda":
                torch.cuda.empty_cache()

    print(f"\nResults -> {path}")
    print(f"Figure:  uv run eval/bench_plots.py --FOLDER {args.FOLDER}")


if __name__ == "__main__":
    main()
