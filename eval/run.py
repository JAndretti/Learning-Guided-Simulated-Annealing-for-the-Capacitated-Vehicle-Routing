import argparse
import importlib
import os
import sys

import torch

# Ensure the eval/ directory is on sys.path so handler modules resolve correctly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

HANDLERS: dict[str, str] = {
    "random": "handler_random",
    "X": "handler_X",
    "X_batch": "handler_X_batch",
    "XL": "handler_XL",
    "XL_batch": "handler_XL_batch",
    "XML": "handler_XML",
}

# --- Pass 1: extract --dataset before full parse (optional so --help works) ---
pre_parser = argparse.ArgumentParser(add_help=False)
pre_parser.add_argument("--dataset", choices=HANDLERS.keys(), default=None)
pre_args, _ = pre_parser.parse_known_args()

handler = importlib.import_module(HANDLERS[pre_args.dataset]) if pre_args.dataset else None

# --- Pass 2: full parse with common + handler-specific args ---
parser = argparse.ArgumentParser(
    description="LGSA Evaluation — single entry point for all benchmarks"
)
parser.add_argument("--dataset", required=True, choices=HANDLERS.keys(), help="Benchmark dataset")
parser.add_argument("--FOLDER", type=str, default="BEST", help="Model folder under wandb/LGSA/")
parser.add_argument(
    "--INIT",
    type=str,
    default="random",
    choices=[
        "random",
        "isolate",
        "sweep",
        "nearest_neighbor",
        "Clark_and_Wright",
        "Clark_and_Wright_reversal",
    ],
)
parser.add_argument("--OUTER_STEPS", type=int, default=10000)
parser.add_argument("--seed", type=int, default=1234)
parser.add_argument(
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
parser.add_argument(
    "--dtype",
    type=str,
    default="float32",
    choices=["float32", "bfloat16", "float16"],
    help="Floating-point precision for the full SA loop",
)

if handler is not None:
    handler.add_args(parser)
args = parser.parse_args()

_DTYPE_MAP = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}
args.torch_dtype = _DTYPE_MAP[args.dtype]

if handler is None:
    parser.error("argument --dataset is required")

handler.run(args)
