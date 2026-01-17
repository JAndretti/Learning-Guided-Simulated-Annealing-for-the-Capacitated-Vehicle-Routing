# ============================================================================
# FAST SIMULATED ANNEALING INFERENCE
# ============================================================================

from typing import Dict, List, Any

import torch
from tqdm import tqdm

from model import SAModel
from problem import CVRP
from utils import extend_to

from .scheduler import Scheduler

# ============================================================================
# UTILITY FUNCTIONS (INLINED OR SIMPLIFIED)
# ============================================================================


def scale_to_unit(
    value: torch.Tensor, min_value: float, max_value: float
) -> torch.Tensor:
    """Scales a [min, max] value to [0, 1] for the neural net input."""
    return (value - min_value) / (max_value - min_value)


def metropolis_accept(
    cost_improvement: torch.Tensor, current_temp: torch.Tensor, device: str
) -> torch.Tensor:
    """
    Vectorized Metropolis-Hastings acceptance.
    """
    # Calculate acceptance probability: exp(gain / T)
    # We clip at 1.0 because if gain > 0 (good move), exp is > 1
    # Note: cost_improvement is (current - proposed), so positive means improvement.

    acceptance_prob = torch.exp(cost_improvement / current_temp)
    # Clamp is faster than minimum with ones_like in some kernels, but minimum is safe
    acceptance_prob = torch.clamp(acceptance_prob, max=1.0)

    random_sample = torch.rand(acceptance_prob.shape, device=device)
    is_accepted = (random_sample < acceptance_prob).long()

    return is_accepted


# ============================================================================
# MAIN INFERENCE LOOP
# ============================================================================


def sa_test(
    actor: SAModel,
    problem: CVRP,
    initial_solution: torch.Tensor,
    config: dict,
    # Unused args kept for interface compatibility if needed, else remove
    baseline: bool = False,
    random_std: float = 0.2,
    greedy: bool = False,
    record_state: bool = False,
    replay_buffer=None,
    train: bool = False,
    epoch: int = 0,
    device: str = "",
    desc_tqdm: str = "SA Inference",
) -> Dict[str, torch.Tensor | List[Any]]:
    """
    High-performance Simulated Annealing Inference Loop.
    Strips out RL rewards, buffers, and logging to maximize throughput.
    """
    if device == "":
        device = str(initial_solution.device)

    # --- 1. SETUP ---
    total_steps = config["OUTER_STEPS"] if train else config["TEST_OUTER_STEPS"]

    # Initialize Scheduler
    scheduler = Scheduler(
        config["SCHEDULER"],
        T_max=config["INIT_TEMP"],
        T_min=config["STOP_TEMP"],
        step_max=total_steps,
    )

    # Initialize Tensors
    batch_size = initial_solution.shape[0]

    # Best found so far
    best_solution = initial_solution.clone()
    best_cost = problem.cost(best_solution)

    # Current state of the chain
    current_solution = initial_solution.clone()
    current_cost = best_cost.clone()

    # Temperature management
    current_temp = torch.full((batch_size,), config["INIT_TEMP"], device=device)

    # Pre-allocate tensor for advancement ratio (scalar that changes per step)
    # We create the full schedule of temperatures in advance to avoid CPU-GPU sync inside loop
    # (Optional optimization, strictly keeping your Scheduler logic here for safety)

    # --- 2. OPTIMIZATION LOOP ---

    # torch.inference_mode() is highly recommended for inference-only loops
    # It disables view tracking and version counters, faster than no_grad()
    with torch.inference_mode():
        # Initial State Construction
        # We need to construct the first state before the loop
        normalized_temp = scale_to_unit(
            current_temp, config["STOP_TEMP"], config["INIT_TEMP"]
        )
        adv = torch.tensor(1.0, device=device)  # Initial progress = 1.0 (start)

        # Build state
        state_components = problem.build_state_components(
            current_solution, normalized_temp, adv
        )
        current_state = problem.to_state(*state_components)

        progress_bar = tqdm(
            range(total_steps),
            desc=desc_tqdm,
            colour="green",
            leave=False,
            disable=not config.get(
                "VERBOSE", True
            ),  # Option to silence tqdm for max speed
        )

        for step in progress_bar:
            # A. Generate Action (Neural Proposal)
            # We discard log_probs and masks since we don't train here
            action, _, _ = actor.sample(current_state, greedy=greedy, problem=problem)

            # B. Apply Action
            # problem.from_state splits the big tensor back into useful components (coords, etc)
            sol_components = problem.from_state(current_state)

            # Proposed solution
            proposed_sol, is_valid = problem.update(sol_components[0], action)
            proposed_cost = problem.cost(proposed_sol)

            # C. Evaluate (Metropolis)
            cost_improvement = current_cost - proposed_cost

            if config["METROPOLIS"]:
                is_accepted = metropolis_accept(cost_improvement, current_temp, device)
            else:
                # Greedy / Hill Climbing mode
                is_accepted = (cost_improvement >= 0).long()

            # D. Update Current State
            # current = accepted ? proposed : current
            # We only update indices where is_accepted is True to save ops?
            # Vectorized 'where' is usually faster than masking on GPU due to coherence.

            # Update Cost
            current_cost = torch.where(is_accepted.bool(), proposed_cost, current_cost)

            # Update Solution Tensor
            # extend_to broadcasts the (B,) flag to (B, N, 1)
            is_accepted_expanded = extend_to(is_accepted, current_solution)
            current_solution = torch.where(
                is_accepted_expanded.bool(), proposed_sol, current_solution
            )

            # Important: Sync problem internal state (demands, masks) to the new current solution
            problem.update_tensor(current_solution)

            # E. Update Best Found
            is_improvement = current_cost < best_cost
            if is_improvement.any():
                best_cost = torch.minimum(current_cost, best_cost)
                is_imp_expanded = extend_to(is_improvement.long(), best_solution)
                best_solution = torch.where(
                    is_imp_expanded.bool(), current_solution, best_solution
                )

            # F. Prepare Next Step (Temperature & State)
            # Update Temperature
            new_temp_scalar = scheduler.step(step)
            current_temp.fill_(new_temp_scalar)  # In-place update is slightly faster

            # Update State Inputs
            # normalized temp for the network
            norm_temp = scale_to_unit(
                current_temp, config["STOP_TEMP"], config["INIT_TEMP"]
            )
            # progress ratio (decreases 1 -> 0)
            adv = torch.tensor(1.0 - (step / total_steps), device=device)

            # Rebuild state tensor for next iteration
            state_components = problem.build_state_components(
                current_solution, norm_temp, adv
            )
            current_state = problem.to_state(*state_components)

            # Clean Cache periodically if using CUDA to prevent fragmentation OOM on huge batches
            # Only do this if strictly necessary as it slows down the loop
            # if step % 100 == 0 and "cuda" in device:
            #     torch.cuda.empty_cache()

    # --- 3. RETURN RESULTS ---

    return {
        "best_x": best_solution,
        "min_cost": best_cost,
        # We can return dummy values for others to avoid breaking external unpacking
        "primal": best_cost,
        "ngain": torch.tensor(0.0),
        "n_acc": torch.tensor(0.0),
        "n_rej": torch.tensor(0.0),
        "distributions": [],
        "is_valid": torch.tensor([]),
        "states": [],
        "actions": [],
        "acceptance": [],
        "costs": [],
        "init_cost": torch.tensor(0.0),  # Calculate if needed, else 0
        "reward": torch.tensor(0.0),
        "all_rewards": torch.tensor(0.0),
        "temperature": [],
        "best_step": torch.tensor(0.0),
        "capacity_left": torch.tensor(0.0),
    }
