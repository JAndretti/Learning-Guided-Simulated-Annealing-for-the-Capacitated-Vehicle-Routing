# eval/

Single-entry-point evaluation for LG-SA. All benchmarks are accessed through `run.py` with a `--dataset` flag.

## Usage

```bash
# Random / Nazari / Uchoa (multiple models in one folder)
uv run eval/run.py --dataset random --FOLDER BEST --dim 100 --DATA nazari

# CVRPLib Set X — sequential, one instance at a time
uv run eval/run.py --dataset X --FOLDER BEST --DATA_PATH bdd/X

# CVRPLib Set X — bucketed batch (heterogeneous sizes, ghost-node padding)
uv run eval/run.py --dataset X_batch --FOLDER BEST --DATA_PATH bdd/X --mode bucket --n_buckets 5

# CVRPLib Set XL — sequential, no optimal cost
uv run eval/run.py --dataset XL --FOLDER BEST --DATA_PATH bdd/XL

# Queiroga XML — single batch (all instances same size)
uv run eval/run.py --dataset XML --FOLDER BEST --DATA_PATH bdd/XML --SOL_PATH bdd/solutions
```

Run `uv run eval/run.py --dataset <name> --help` for the full argument list of any handler.

## Common arguments

| Arg | Default | Description |
|-----|---------|-------------|
| `--dataset` | required | `{random, X, X_batch, XL, XML}` |
| `--FOLDER` | `BEST` | Model folder name under `wandb/LGSA/` |
| `--INIT` | `random` | Init heuristic |
| `--OUTER_STEPS` | `10000` | SA steps |
| `--seed` | `1234` | Random seed |
| `--device` | auto | `cpu` / `cuda` / `mps` |

Results are saved to `res/<FOLDER>/` with automatic versioning (e.g. `cvrplib_X_results.csv` → `cvrplib_X_results_2.csv`).

## Structure

```
eval/
├── run.py              # CLI entry point — two-pass argparse, dispatches to handler
├── costs.py            # cvrplib_rounded_cost, exact_euclidean_cost, extract_and_cost
├── eval_io.py          # find_model(s), load_vrp_instance, save_results, HP utilities
├── solver.py           # build_actor, build_problem, run_lgsa, set_seed, warmup_cuda
├── handler_random.py   # random/Nazari/Uchoa: loops over multiple checkpoints
├── handler_X.py        # CVRPLib Set X: sequential, rounded integer cost
├── handler_XL.py       # CVRPLib Set XL: sequential, no .sol files
├── handler_XML.py      # Queiroga XML: single batch, exact float cost
└── handler_X_batch.py  # CVRPLib Set X: bucketed batching, ghost-node padding
```

## Adding a new benchmark

1. Create `eval/handler_<name>.py` with two functions:

```python
def add_args(parser: argparse.ArgumentParser) -> None:
    # register any dataset-specific CLI args

def run(args: argparse.Namespace) -> None:
    # load model, run eval, save results
```

2. Register it in `run.py`:

```python
HANDLERS: dict[str, str] = {
    ...
    "<name>": "handler_<name>",
}
```

That's it — no other files need to change.

## Cost conventions

| Dataset | Function | Convention |
|---------|----------|------------|
| Set X, XL | `cvrplib_rounded_cost` | Euclidean distance **rounded to nearest integer** per edge |
| Queiroga XML | `exact_euclidean_cost` | Exact float Euclidean distance |
| Set X batch (padded) | `extract_and_cost` | Ghost-node cleanup → `cvrplib_rounded_cost` |
