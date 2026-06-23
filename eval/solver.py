import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from init import inf_test_model, initialize_models
from problem import CVRP


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def warmup_cuda() -> None:
    if torch.cuda.is_available():
        torch.empty(1, device="cuda").uniform_()
        torch.cuda.synchronize()


def _extract_loss(filename: str) -> float:
    try:
        return float(filename.split("_")[-1][:-3])
    except ValueError:
        return float("inf")


def _load_weights(model: nn.Module, folder: str) -> nn.Module:
    files = [f for f in os.listdir(folder) if f.endswith(".pt") and "actor" in f]
    if files:
        best_file = min(files, key=_extract_loss)
        model.load_state_dict(
            torch.load(
                os.path.join(folder, best_file),
                weights_only=True,
                map_location="cpu",
            )
        )
    return model


def build_actor(
    HP: dict,
    model_path: str,
    input_dim: int,
    device: str,
    seed: int,
    dtype: torch.dtype = torch.float32,
) -> nn.Module:
    """
    Build and return the actor with loaded weights in eval mode.

    model_path is the model directory (passed to _load_weights).
    """
    actor, _ = initialize_models(
        model_type=HP["MODEL"],
        critic_type="ff",
        embedding_dim=HP["EMBEDDING_DIM"],
        entry=input_dim,
        num_h_layers=HP["NUM_H_LAYERS"],
        update_method=HP["UPDATE_METHOD"],
        heuristic=HP["HEURISTIC"],
        seed=seed,
        device=device,
    )
    actor = _load_weights(actor, model_path)
    actor = actor.to(device)
    actor = actor.to(dtype)
    actor.eval()
    return actor


def build_problem(HP: dict, dim: int, n_problems: int, device: str) -> CVRP:
    """
    Create a CVRP with heuristic and feature flags set.

    The returned problem has no data yet — call problem.generate_params() next.
    """
    problem = CVRP(dim=dim, n_problems=n_problems, device=device, params=HP)
    problem.manual_seed(HP.get("SEED", 0))
    problem.set_heuristic(HP["HEURISTIC"])
    problem.set_feature_flags(HP["features"])
    return problem


def run_lgsa(
    actor: nn.Module,
    problem: CVRP,
    init_x: torch.Tensor,
    HP: dict,
    *,
    outer_steps: int,
    baseline: bool = False,
    greedy: bool = False,
    dtype: torch.dtype = torch.float32,
) -> dict:
    """
    Run LGSA inference.

    Sets HP["TEST_OUTER_STEPS"] on a copy of HP to avoid mutating the caller's dict.
    Returns the result dict from inf_test_model unchanged.
    """
    hp = {**HP, "TEST_OUTER_STEPS": outer_steps}
    results_td, _ = inf_test_model(
        actor=actor,
        problem=problem,
        initial_solutions=init_x,
        config=hp,
        baseline=baseline,
        greedy=greedy,
        dtype=dtype,
    )
    return results_td


# 8 isometries of the unit square (dihedral group D4).
# All preserve pairwise Euclidean distances: ‖aug(pᵢ)−aug(pⱼ)‖ = ‖pᵢ−pⱼ‖.
AUGMENTATIONS = [
    lambda p: p,  # identity
    lambda p: torch.stack([1 - p[..., 0], p[..., 1]], dim=-1),  # flip x
    lambda p: torch.stack([p[..., 0], 1 - p[..., 1]], dim=-1),  # flip y
    lambda p: torch.stack([1 - p[..., 0], 1 - p[..., 1]], dim=-1),  # flip both
    lambda p: torch.stack([p[..., 1], p[..., 0]], dim=-1),  # swap xy
    lambda p: torch.stack([1 - p[..., 1], p[..., 0]], dim=-1),  # swap + flip x
    lambda p: torch.stack([p[..., 1], 1 - p[..., 0]], dim=-1),  # swap + flip y
    lambda p: torch.stack([1 - p[..., 1], 1 - p[..., 0]], dim=-1),  # swap + flip both
]


def augment_coords(coords: torch.Tensor, k: int) -> torch.Tensor:
    """Apply the k-th dihedral augmentation to a coordinate tensor of shape [..., 2]."""
    return AUGMENTATIONS[k](coords)
