"""Speed-focused LG-SA inference for an external (Rust) solver.

This is a thin, parameterized CLI wrapper around the lightweight `sa_test` loop.
Instances are *not* generated here — the caller passes coordinates, demands,
vehicle capacity and an initial solution through a NumPy ``.npz`` file, and the
improved solution is written back to another ``.npz``.

Two ways to drive it from the Rust side:

  * One-shot (reloads the model each call; fine for a single big batch):
        uv run example/fast_ex.py --input in.npz --output out.npz --outer-steps 1000

  * Persistent server (loads the model once; best for many repeated calls):
        uv run example/fast_ex.py --serve
    then write one JSON request per line to stdin and read one JSON reply per
    line from stdout. See serve() for the protocol.

Input ``.npz`` arrays (N instances, DIM customers)
--------------------------------------------------
    coords    float32 [N, DIM + 1, 2]   depot at index 0, customers 1..DIM
    demands   int64   [N, DIM + 1]      depot demand is 0
    capacity  int64   [N]  or [N, 1]    per-instance vehicle capacity
    init      int64   [N, L]            initial solution, one row per instance:
                                        customer indices 1..DIM with 0 marking a
                                        return to the depot / route separator,
                                        zero-padded on the right. Example row:
                                        [3, 1, 7, 0, 2, 5, 0, 4, 6, 0, 0, ...]
                                        encodes routes [3,1,7], [2,5], [4,6].

Output ``.npz`` arrays
----------------------
    best_routes int64   [N, L']  best solution found, same flat 0-delimited
                                 layout as ``init`` (starts with 0, 0-padded).
    cost        float32 [N]      length of best_routes per instance.
    init_cost   float32 [N]      length of the provided initial solution.
"""

import argparse
import contextlib
import glob
import json
import os
import random
import sys

import numpy as np
import torch

EXAMPLE_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.abspath(os.path.join(EXAMPLE_DIR, "..", "src")))
import yaml  # noqa: E402

from algo.heur_init import MULT  # noqa: E402
from model import CVRPActor  # noqa: E402
from problem import CVRP  # noqa: E402
from sa import sa_test  # noqa: E402
from utils import is_feasible  # noqa: E402


def set_seed(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_actor(model_path: str, input_dim: int, hp: dict, device: str) -> CVRPActor:
    """Build the actor from the checkpoint's stored hyperparameters (mirrors src/init.py)."""
    actor = CVRPActor(
        embed_dim=hp["EMBEDDING_DIM"],
        c=input_dim,
        num_hidden_layers=hp["NUM_H_LAYERS"],
        device=device,
        method=hp["UPDATE_METHOD"],
        cond_rank=hp.get("COND_RANK", False),
        cond_detour=hp.get("COND_DETOUR", False),
        global_context=hp.get("GLOBAL_CONTEXT", False),
        logit_clip=hp.get("LOGIT_CLIP", 0.0),
        learnable_temp=hp.get("LEARNABLE_TEMP", False),
    )
    checkpoint = sorted(glob.glob(os.path.join(model_path, "*.pt")))[0]
    actor.load_state_dict(
        torch.load(checkpoint, map_location=torch.device("cpu"), weights_only=True)
    )
    actor.to(device)
    actor.eval()
    return actor


def load_hp(model_path: str) -> dict:
    """Read the flat config dict stored alongside the checkpoint (see example.py)."""
    with open(os.path.join(model_path, "HP.yaml")) as f:
        content = f.read().replace("!!python/object:setup.HP._HP", "")
    return yaml.safe_load(content)["config"]


def routes_to_solution(init_rows: np.ndarray, dim: int) -> torch.Tensor:
    """Convert flat, 0-delimited init rows into the canonical solution tensor.

    Splits every row on 0 to recover the routes, checks that customers 1..DIM
    each appear exactly once, then rebuilds a depot-delimited sequence
    ``[0, r1..., 0, r2..., 0, ..., 0]`` padded to a common length. The internal
    format is ``[N, L, 1]`` (long), starting and ending with the depot.
    """
    expected = set(range(1, dim + 1))
    num_nodes = dim + 1
    base_len = num_nodes + int(num_nodes * MULT)  # length used at training time

    seqs: list[list[int]] = []
    for i, row in enumerate(init_rows):
        routes: list[list[int]] = []
        current: list[int] = []
        for v in row.tolist():
            v = int(v)
            if v == 0:
                if current:
                    routes.append(current)
                    current = []
            else:
                current.append(v)
        if current:
            routes.append(current)

        visited = [c for r in routes for c in r]
        seen = set(visited)
        if len(visited) != len(seen):
            raise ValueError(f"instance {i}: init visits a customer more than once")
        if seen != expected:
            missing = sorted(expected - seen)
            extra = sorted(seen - expected)
            raise ValueError(
                f"instance {i}: init must visit customers 1..{dim} exactly once "
                f"(missing={missing[:8]}, unexpected={extra[:8]})"
            )

        seq = [0]
        for r in routes:
            seq.extend(r)
            seq.append(0)
        seqs.append(seq)

    length = max(base_len, max(len(s) for s in seqs))
    out = torch.zeros(len(seqs), length, dtype=torch.long)
    for i, s in enumerate(seqs):
        out[i, : len(s)] = torch.tensor(s, dtype=torch.long)
    return out.unsqueeze(-1)


def build_input_dim(hp: dict, device: str) -> int:
    """Actor input width depends only on the feature flags, not on N or DIM."""
    probe = CVRP(dim=1, n_problems=1, device=device, params={})
    probe.set_feature_flags(hp["features"])
    return probe.get_input_dim()


def solve(
    coords: torch.Tensor,
    demands: torch.Tensor,
    capacity: torch.Tensor,
    init_solution: torch.Tensor,
    hp: dict,
    *,
    actor: CVRPActor | None = None,
    model_path: str | None = None,
    device: str = "cpu",
    outer_steps: int = 1000,
    greedy: bool = False,
    desc: str = "LG-SA",
) -> dict[str, torch.Tensor]:
    """Run LG-SA (sa_test) on one batch of instances. Tensors are [N, ...].

    Pass a preloaded ``actor`` (serve mode) to skip the checkpoint load; otherwise
    ``model_path`` must be given and the actor is built here (one-shot mode). The
    actor is instance-agnostic, so the same one is reused across any N / DIM.
    """
    n_problems, num_nodes = demands.shape
    dim = num_nodes - 1

    cfg = dict(hp)
    cfg.update(
        {
            "PROBLEM_DIM": dim,
            "N_PROBLEMS": n_problems,
            "DEVICE": device,
            "TEST_OUTER_STEPS": outer_steps,
            "INIT": cfg.get("INIT", "random"),
        }
    )

    problem = CVRP(dim=dim, n_problems=n_problems, device=device, params=cfg)
    problem.set_heuristic(cfg["HEURISTIC"])
    problem.set_feature_flags(cfg["features"])
    problem.generate_params(coords, demands, capacity)

    if actor is None:
        if model_path is None:
            raise ValueError("solve() needs either a preloaded actor or a model_path")
        actor = load_actor(model_path, problem.get_input_dim(), cfg, device)

    init_solution = init_solution.to(device)
    feasible = is_feasible(init_solution, problem.get_demands(init_solution), problem.capacity)
    if not feasible.all():
        bad = torch.nonzero(~feasible).flatten().tolist()
        raise ValueError(f"init solution violates capacity on instances: {bad[:16]}")

    init_solution = problem.init_parameters(init_solution)
    init_cost = problem.cost(init_solution)

    with torch.no_grad():
        res, _ = sa_test(
            actor, problem, init_solution, cfg, greedy=greedy, desc_tqdm=desc
        )

    return {
        "best_x": res["best_x"].squeeze(-1).cpu(),  # [N, L]
        "cost": res["min_cost"].cpu(),  # [N]
        "init_cost": init_cost.cpu(),  # [N]
    }


def read_instance(path: str, device: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load one .npz instance file into (coords, demands, capacity, init_solution)."""
    data = np.load(path)
    coords = torch.as_tensor(np.asarray(data["coords"]), dtype=torch.float32, device=device)
    demands = torch.as_tensor(np.asarray(data["demands"]), dtype=torch.long, device=device)
    capacity = torch.as_tensor(np.asarray(data["capacity"]), dtype=torch.long, device=device)
    if capacity.dim() == 1:
        capacity = capacity.unsqueeze(-1)  # [N] -> [N, 1]
    dim = demands.shape[1] - 1
    init_solution = routes_to_solution(np.asarray(data["init"]), dim)
    return coords, demands, capacity, init_solution


def write_result(path: str, result: dict[str, torch.Tensor]) -> None:
    np.savez(
        path,
        best_routes=result["best_x"].numpy().astype(np.int64),
        cost=result["cost"].numpy().astype(np.float32),
        init_cost=result["init_cost"].numpy().astype(np.float32),
    )


def run_one_shot(args, model_path: str, hp: dict) -> None:
    coords, demands, capacity, init_solution = read_instance(args.input, args.device)
    result = solve(
        coords=coords,
        demands=demands,
        capacity=capacity,
        init_solution=init_solution,
        hp=hp,
        model_path=model_path,
        device=args.device,
        outer_steps=args.outer_steps,
        greedy=args.greedy,
    )
    write_result(args.output, result)
    print(
        f"solved {demands.shape[0]} instance(s): "
        f"init {result['init_cost'].mean():.3f} -> best {result['cost'].mean():.3f} "
        f"-> {args.output}",
        file=sys.stderr,
    )


def serve(args, model_path: str, hp: dict) -> None:
    """Persistent request loop: load the model once, then answer requests forever.

    Protocol (line-oriented, so stdout carries exactly one JSON object per reply):
      * Load the checkpoint, then emit {"status": "ready"} on stdout.
      * Read one JSON request per line from stdin, e.g.
            {"input": "in.npz", "output": "out.npz", "outer_steps": 1000, "greedy": false}
        `input`/`output` are required; `outer_steps`/`greedy` are optional per call.
        Send {"cmd": "shutdown"} (or close stdin) to stop.
      * Reply on stdout with one JSON line:
            {"status": "ok", "output": "out.npz", "n": 4,
             "init_cost_mean": ..., "cost_mean": ...}
        or {"status": "error", "message": "..."} on a bad request.
    stderr may carry progress bars / warnings; stdout is reserved for the protocol.
    """
    device = args.device
    actor = load_actor(model_path, build_input_dim(hp, device), hp, device)
    devnull = open(os.devnull, "w")  # reused to keep stdout clean during solve()

    def reply(obj: dict) -> None:
        sys.stdout.write(json.dumps(obj) + "\n")
        sys.stdout.flush()

    reply({"status": "ready", "device": device, "model": os.path.basename(model_path)})

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError as exc:
            reply({"status": "error", "message": f"invalid JSON request: {exc}"})
            continue

        if req.get("cmd") == "shutdown":
            break

        try:
            coords, demands, capacity, init_solution = read_instance(req["input"], device)
            # Keep stdout clean: swallow tqdm/warnings to stderr's devnull during solve.
            with contextlib.redirect_stderr(devnull):
                result = solve(
                    coords=coords,
                    demands=demands,
                    capacity=capacity,
                    init_solution=init_solution,
                    hp=hp,
                    actor=actor,
                    device=device,
                    outer_steps=int(req.get("outer_steps", args.outer_steps)),
                    greedy=bool(req.get("greedy", args.greedy)),
                )
            write_result(req["output"], result)
            reply(
                {
                    "status": "ok",
                    "output": req["output"],
                    "n": int(demands.shape[0]),
                    "init_cost_mean": float(result["init_cost"].mean()),
                    "cost_mean": float(result["cost"].mean()),
                }
            )
        except Exception as exc:  # report, but keep serving
            reply({"status": "error", "message": f"{type(exc).__name__}: {exc}"})


def main() -> None:
    parser = argparse.ArgumentParser(description="LG-SA inference from .npz instance files.")
    parser.add_argument("--serve", action="store_true", help="Persistent stdin/stdout request loop (load model once).")
    parser.add_argument("--input", help="One-shot mode: input .npz (coords, demands, capacity, init).")
    parser.add_argument("--output", help="One-shot mode: output .npz (best_routes, cost, init_cost).")
    parser.add_argument("--model", default=None, help="Model dir (default: the one under example/models).")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--outer-steps", type=int, default=1000, help="Default SA steps (per-request override in serve mode).")
    parser.add_argument("--greedy", action="store_true", help="Default greedy sampling (per-request override in serve mode).")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    set_seed(args.seed)
    model_path = args.model or sorted(glob.glob(os.path.join(EXAMPLE_DIR, "models", "*")))[0]
    hp = load_hp(model_path)

    if args.serve:
        serve(args, model_path, hp)
    else:
        if not args.input or not args.output:
            parser.error("--input and --output are required unless --serve is given")
        run_one_shot(args, model_path, hp)


if __name__ == "__main__":
    main()
