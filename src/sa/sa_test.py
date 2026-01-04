from typing import Dict, Optional

import torch
from tqdm import tqdm

from model import SAModel
from problem import Problem
from utils import extend_to

from .scheduler import Scheduler

# ================================
# HELPER FUNCTIONS
# ================================


def scale_between(
    value: torch.Tensor, min_value: float, max_value: float
) -> torch.Tensor:
    return min_value + (max_value - min_value) * value


def scale_to_unit(
    value: torch.Tensor, min_value: float, max_value: float
) -> torch.Tensor:
    return (value - min_value) / (max_value - min_value)


def metropolis_accept(
    cost_improvement: torch.Tensor, current_temp: torch.Tensor
) -> torch.Tensor:
    """Returns 1 if accepted, 0 if rejected."""
    # min(1, exp(gain/temp))
    acceptance_prob = torch.minimum(
        torch.exp(cost_improvement / current_temp), torch.ones_like(cost_improvement)
    )
    random_sample = torch.rand(acceptance_prob.shape, device=cost_improvement.device)
    return (random_sample < acceptance_prob).long()


# ================================
# OPTIMIZATION LOOP
# ================================


def sa_test(
    actor: SAModel,
    problem: Problem,
    initial_solution: torch.Tensor,
    config: dict,
    baseline: bool = False,
    greedy: bool = False,
    device: Optional[torch.device] = None,
    desc_tqdm: str = "SA Optimization",
) -> Dict[str, torch.Tensor]:
    """
    Ultra-lightweight Simulated Annealing.
    Tracks ONLY the best cost found. No solution history or auxiliary metrics.
    """

    # 1. Setup
    if device is None:
        device = initial_solution.device

    batch_size = initial_solution.shape[0]

    scheduler = Scheduler(
        config["SCHEDULER"],
        T_max=config["INIT_TEMP"],
        T_min=config["STOP_TEMP"],
        step_max=config["OUTER_STEPS"],
    )

    # 2. Initialize State
    # current_solution is needed to navigate the search space
    current_solution = initial_solution.clone()
    current_cost = problem.cost(current_solution)

    # We only track the lowest cost seen so far
    best_cost = current_cost.clone()

    # Initialize Temperature
    current_temp = torch.ones(batch_size, device=device)
    current_temp = scale_between(current_temp, config["STOP_TEMP"], config["INIT_TEMP"])

    # Initial Neural State
    normalized_temp = scale_to_unit(
        current_temp, config["STOP_TEMP"], config["INIT_TEMP"]
    )
    advancement = torch.tensor(1.0, device=device)

    components = problem.build_state_components(
        current_solution, normalized_temp, advancement
    )
    current_state = problem.to_state(*components).to(device)

    # 3. Main Loop
    with torch.inference_mode():
        for step in tqdm(
            range(config["OUTER_STEPS"]), desc=desc_tqdm, colour="green", leave=False
        ):
            for inner_step in range(config["INNER_STEPS"]):
                # A. Generate Action
                with torch.amp.autocast(device.type):
                    if baseline:
                        action, _, _ = actor.baseline_sample(
                            current_state, problem=problem
                        )
                    else:
                        action, _, _ = actor.sample(
                            current_state,
                            greedy=greedy,
                            problem=problem,
                        )

                # B. Proposed Update
                solution_components, *_ = problem.from_state(current_state)
                proposed_solution, _ = problem.update(solution_components, action)
                proposed_cost = problem.cost(proposed_solution)

                # C. Acceptance
                cost_diff = current_cost - proposed_cost

                if config["METROPOLIS"]:
                    is_accepted = metropolis_accept(cost_diff, current_temp)
                else:
                    is_accepted = torch.ones_like(cost_diff, dtype=torch.long)

                # D. Update Current State
                # New Cost
                current_cost = (
                    is_accepted * proposed_cost + (1 - is_accepted) * current_cost
                )

                # New Solution Configuration
                is_accepted_expanded = extend_to(is_accepted, proposed_solution)
                current_solution = (
                    is_accepted_expanded * proposed_solution
                    + (1 - is_accepted_expanded) * current_solution
                ).long()

                # E. Update Best Cost (Min operation is very cheap)
                best_cost = torch.minimum(current_cost, best_cost)

                # F. Prepare Next Step (Skip on last inner step to save compute)
                if inner_step == config["INNER_STEPS"] - 1:
                    current_temp = scheduler.step(step).to(device).repeat(batch_size)
                    problem.update_tensor(current_solution)

                    normalized_temp = scale_to_unit(
                        current_temp, config["STOP_TEMP"], config["INIT_TEMP"]
                    )
                    adv = torch.tensor(
                        1 - ((step + 1) / config["OUTER_STEPS"]), device=device
                    )

                    next_components = problem.build_state_components(
                        current_solution, normalized_temp, adv
                    )
                    current_state = problem.to_state(*next_components).to(device)

    # 4. Final Cleanup
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {"min_cost": best_cost}
