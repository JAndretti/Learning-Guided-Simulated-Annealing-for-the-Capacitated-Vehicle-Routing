"""
GPU batch-scaling sweep: total runtime and per-instance runtime vs. batch size.
(Old paper: Table "Impact of batch size on GPU execution time" + Figure "GPU
Batched Execution Scaling".)

Fixed N and step budget; only the number of instances solved in parallel varies.
The interesting structure is the three regimes — under-saturated (constant-ish
total time), peak efficiency, then memory-bound linear scaling.

Each batch size gets its own problem build, since the batch dimension is baked
into the CVRP tensors. A discarded warm-up run precedes the timed sweep so the
first measured point does not absorb cuDNN/allocator initialisation.

Usage
-----
    uv run eval/bench_batch.py --FOLDER BEST --dim 100 --steps 10000 \
        --batches 100,500,1000,2000,3000,4000,5000,10000

Output: res/<FOLDER>/bench/batch.csv  (one row per batch size)
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

KEYS = ["dim", "steps", "batch_size", "dtype"]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_common_args(p)
    p.add_argument("--dim", type=int, default=100)
    p.add_argument("--steps", type=int, default=10000)
    p.add_argument(
        "--batches",
        type=int_list,
        default="100,500,1000,2000,3000,4000,5000,6000,7000,8000,9000,10000",
    )
    p.add_argument(
        "--repeats", type=int, default=1, help="Timed repetitions per batch size (median reported)"
    )
    return p


def main() -> None:
    args = build_parser().parse_args()
    prepare_run(args)
    dtype = DTYPE_MAP[args.dtype]

    if args.device != "cuda":
        print(f"warning: device={args.device} — the batch-scaling regimes are a GPU phenomenon")

    path = bench_path(args.FOLDER, out_name("batch", args.tag))
    done = load_done(path, KEYS) if args.resume else set()

    todo = [b for b in args.batches if (args.dim, args.steps, b, args.dtype) not in done]
    if not todo:
        print("All batch sizes present, nothing to do.")
        return

    # Warm-up on the smallest batch: allocator growth and kernel autotuning would
    # otherwise land entirely on the first timed point.
    warm_actor, warm_problem, warm_x, warm_HP, _ = build_problem_and_actor(
        args.FOLDER,
        args.dim,
        min(todo),
        min(args.steps, 200),
        args.device,
        args.seed,
        dtype=dtype,
        init=args.INIT,
        data=args.DATA,
    )
    timed_lgsa(
        warm_actor, warm_problem, warm_x, warm_HP, min(args.steps, 200), args.device, dtype=dtype
    )
    del warm_actor, warm_problem, warm_x
    if args.device == "cuda":
        torch.cuda.empty_cache()

    for batch in tqdm(todo, desc="batch sizes", leave=False):
        actor, problem, init_x, HP, setup_time = build_problem_and_actor(
            args.FOLDER,
            args.dim,
            batch,
            args.steps,
            args.device,
            args.seed,
            dtype=dtype,
            init=args.INIT,
            data=args.DATA,
        )
        if args.device == "cuda":
            torch.cuda.reset_peak_memory_stats()

        times, cost = [], float("nan")
        for _ in range(args.repeats):
            cost, sa_time = timed_lgsa(
                actor, problem, init_x, HP, args.steps, args.device, dtype=dtype
            )
            times.append(sa_time)
        sa_time = sorted(times)[len(times) // 2]

        peak_mb = (
            torch.cuda.max_memory_allocated() / 1024**2 if args.device == "cuda" else float("nan")
        )
        append_row(
            path,
            {
                "dim": args.dim,
                "steps": args.steps,
                "batch_size": batch,
                "dtype": args.dtype,
                "cost": cost,
                "sa_time": sa_time,
                "time_per_instance_ms": 1000.0 * sa_time / batch,
                "throughput_inst_per_s": batch / sa_time,
                "peak_mem_MB": peak_mb,
                "setup_time": setup_time,
                "repeats": args.repeats,
                "device": args.device,
            },
        )
        tqdm.write(
            f"  B={batch:>6,} time={sa_time:8.2f}s "
            f"({1000.0 * sa_time / batch:7.2f} ms/inst, peak {peak_mb:.0f} MB) cost={cost:.4f}"
        )

        del actor, problem, init_x
        if args.device == "cuda":
            torch.cuda.empty_cache()

    print(f"\nResults -> {path}")


if __name__ == "__main__":
    main()
