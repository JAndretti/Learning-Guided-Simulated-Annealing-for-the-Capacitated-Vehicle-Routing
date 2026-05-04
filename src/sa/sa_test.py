from typing import Dict, Tuple

import torch
from tqdm import tqdm

from model import SAModel
from problem import CVRP
from utils import extend_to

from .scheduler import Scheduler

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================


def scale_between(
    value: torch.Tensor, min_value: float, max_value: float
) -> torch.Tensor:
    return min_value + (max_value - min_value) * value


def scale_to_unit(
    value: torch.Tensor, min_value: float, max_value: float
) -> torch.Tensor:
    return (value - min_value) / (max_value - min_value)


def normalize(
    actual_improvement: torch.Tensor, initial_cost: torch.Tensor
) -> torch.Tensor:
    MAX_EXPECTED_REL_IMPROVEMENT = 0.2
    relative_improvement = actual_improvement / initial_cost
    return torch.clamp(relative_improvement / MAX_EXPECTED_REL_IMPROVEMENT, -1.0, 1.0)


# ============================================================================
# CORE LOGIC
# ============================================================================


def metropolis_accept(
    cost_improvement: torch.Tensor, current_temp: torch.Tensor, device: str
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Standard Metropolis-Hastings acceptance.
    """
    acceptance_prob = torch.minimum(
        torch.exp(cost_improvement / current_temp), torch.ones_like(cost_improvement)
    )
    random_sample = torch.rand(acceptance_prob.shape, device=device)
    is_accepted = (random_sample < acceptance_prob).long()

    # Zero out improvement for rejected moves to avoid mixing calculation issues later
    actual_improvement = cost_improvement * is_accepted
    return is_accepted, actual_improvement


def calculate_reward(
    config: dict,
    actual_improvement: torch.Tensor,
    is_valid: torch.Tensor,
    best_cost: torch.Tensor,
    old_best_cost: torch.Tensor,
    new_cost: torch.Tensor,
    cumulative_cost: torch.Tensor,
    initial_cost: torch.Tensor,
    step: int,
    total_steps: int,
    epoch: int,
    last_step: bool = False,
) -> torch.Tensor:
    """
    Calculates RL reward. Only called if training is active.
    """
    if config["METHOD"] != "ppo":
        return torch.zeros_like(actual_improvement).view(-1, 1)

    # 1. Weights
    # If we want a transition within the SA run
    progress = step / total_steps if total_steps > 0 else 0.0

    # Optional warmup across epochs (if needed)
    warmup_epochs = config.get("WARMUP_EPOCHS", 0)
    if warmup_epochs > 0 and epoch < warmup_epochs:
        target_weight = epoch / warmup_epochs
        immediate_weight = 1.0 - target_weight
    else:
        target_weight = 1.0
        immediate_weight = 0.0

    # 2. Immediate
    reward_immediate = torch.zeros_like(actual_improvement).view(-1, 1)
    if immediate_weight > 0 or config["REWARD"] in [
        "immediate",
        "hybrid",
        "curriculum",
    ]:
        val_imm = (
            normalize(actual_improvement, initial_cost)
            if config["NORMALIZE_REWARD"]
            else actual_improvement
        )
        reward_immediate = torch.where(
            ~is_valid.bool().squeeze(-1),
            -1.5,
            torch.where(actual_improvement == 0, 0.0, val_imm),
        ).view(-1, 1)

    # 3. Target
    reward_target = torch.zeros_like(actual_improvement).view(-1, 1)
    target_mode = config["REWARD"]

    if target_mode == "dact" or target_mode == "global_best":
        current_step_min = torch.min(new_cost, old_best_cost)
        raw_reward = old_best_cost - current_step_min
        reward_target = torch.where(
            is_valid.bool().view(-1, 1),
            raw_reward.view(-1, 1),
            torch.zeros_like(raw_reward).view(-1, 1),
        ) * config.get("REWARD_SCALE", 10.0)
    elif target_mode == "hybrid":
        current_step_min = torch.min(new_cost, old_best_cost)
        raw_global = old_best_cost - current_step_min
        global_reward = torch.where(
            is_valid.bool().view(-1, 1),
            raw_global.view(-1, 1),
            torch.zeros_like(raw_global).view(-1, 1),
        ) * config.get("REWARD_SCALE", 1.0)

        alpha = config.get("HYBRID_ALPHA", 0.5)
        # Combine weighted immediate and global_best
        reward_target = alpha * reward_immediate + (1 - alpha) * global_reward
    elif target_mode == "curriculum":
        current_step_min = torch.min(new_cost, old_best_cost)
        raw_global = old_best_cost - current_step_min
        global_reward = torch.where(
            is_valid.bool().view(-1, 1),
            raw_global.view(-1, 1),
            torch.zeros_like(raw_global).view(-1, 1),
        ) * config.get("REWARD_SCALE", 1.0)

        # Linear transition from immediate to global
        reward_target = (1 - progress) * reward_immediate + progress * global_reward
    elif target_mode == "curriculum_terminal":
        # Transition from immediate to terminal
        terminal_part = torch.zeros_like(reward_immediate)
        if last_step:
            terminal_part = ((initial_cost - best_cost) / initial_cost).view(-1, 1)

        reward_target = (1 - progress) * reward_immediate + progress * terminal_part
    elif target_mode == "immediate":
        reward_target = reward_immediate
    elif target_mode == "min_cost":
        reward_target = ((initial_cost + best_cost) / initial_cost).view(-1, 1)
    elif target_mode == "primal":
        reward_target = -cumulative_cost.view(-1, 1)

    # Combine with warmup weights if any
    final_reward = (immediate_weight * reward_immediate) + (
        target_weight * reward_target
    )

    if config.get("REWARD_VALID", False):
        final_reward[~is_valid.view(-1, 1)] = -1.0
    if config.get("REWARD_LAST", False) and last_step:
        final_reward = (
            config.get("REWARD_LAST_SCALE", 1.0)
            * ((initial_cost - best_cost) / initial_cost)
        ).view(-1, 1)

    return final_reward


# ============================================================================
# LIGHTWEIGHT
# ============================================================================


def sa_test(
    actor: SAModel,
    problem: CVRP,
    initial_solution: torch.Tensor,
    config: dict,
    baseline: bool = False,
    random_std: float = 0.2,  # Kept for signature compatibility
    greedy: bool = False,
    record_state: bool = False,  # Ignored in lightweight mode
    replay_buffer=None,
    train: bool = False,
    epoch: int = 0,
    device: str = "",
    desc_tqdm: str = "Simulated Annealing Progress",
) -> Dict[str, torch.Tensor | None | float | float]:

    if device == "":
        device = str(initial_solution.device)

    total_steps = config["OUTER_STEPS"] if train else config["TEST_OUTER_STEPS"]

    # Scheduler
    scheduler = Scheduler(
        config["SCHEDULER"],
        T_max=config["INIT_TEMP"],
        T_min=config["STOP_TEMP"],
        step_max=total_steps,
    )

    # --- Initialization (Minimal) ---
    # We strip out the dictionary 'opt_state' to avoid dict lookup overhead in tight loops
    # and keep variables strictly local.

    current_solution = initial_solution.clone()
    best_solution = initial_solution.clone()

    current_cost = problem.cost(initial_solution)
    best_cost = current_cost.clone()
    initial_cost = current_cost.clone()

    # Needed only for specific reward calculations
    cumulative_cost = (
        torch.ones_like(best_cost)
        if (replay_buffer is not None and config["REWARD"] == "primal")
        else None
    )

    # Temperature Setup
    current_temp = torch.tensor([1.0], device=device).repeat(current_cost.shape[0])
    current_temp = scale_between(current_temp, config["STOP_TEMP"], config["INIT_TEMP"])

    # Initial State
    normalized_temp = scale_to_unit(
        current_temp, config["STOP_TEMP"], config["INIT_TEMP"]
    )
    if baseline:
        current_state = current_solution
    else:
        current_state = problem.to_state(
            *problem.build_state_components(
                current_solution,
                normalized_temp,
                torch.tensor(1.0, device=device),
            )
        ).to(device)

    # Loop
    progress_bar = tqdm(
        range(total_steps),
        desc=("Train/ " if train else "Test/ ") + desc_tqdm,
        colour="green",
        leave=False,
    )

    for step in progress_bar:
        # 1. Action
        with torch.no_grad():
            if baseline:
                action, action_log_prob, mask = actor.baseline_sample(
                    current_state, problem=problem
                )
            else:
                action, action_log_prob, mask = actor.sample(
                    current_state, greedy=greedy, problem=problem
                )

        # 2. Update & Evaluate
        sol_components, *_ = problem.from_state(current_state)
        proposed_sol, is_valid = problem.update(sol_components, action)
        proposed_cost = problem.cost(proposed_sol)

        cost_improvement = current_cost - proposed_cost

        # 3. Metropolis
        if config["METROPOLIS"]:
            is_accepted, actual_improvement = metropolis_accept(
                cost_improvement, current_temp, device
            )
        else:
            is_accepted = torch.ones_like(cost_improvement)
            actual_improvement = cost_improvement

        # 4. Update Current
        # Update cost only where accepted
        current_cost = is_accepted * proposed_cost + (1 - is_accepted) * current_cost

        # Update solution only where accepted
        is_accepted_expanded = extend_to(is_accepted, sol_components)
        current_solution = (
            is_accepted_expanded * proposed_sol
            + (1 - is_accepted_expanded) * sol_components
        ).long()

        # Sync problem internal state
        problem.update_tensor(current_solution)

        # 5. Update Best
        old_best_cost = (
            best_cost.clone() if replay_buffer is not None else best_cost
        )  # Optim: no clone if not needed for reward

        # Check if current cost is better than global best
        is_improvement = (current_cost < best_cost).long()

        # Update best cost
        best_cost = torch.minimum(current_cost, best_cost)

        # Update best solution
        is_imp_expanded = extend_to(is_improvement, current_solution)
        best_solution = (
            is_imp_expanded * current_solution + (1 - is_imp_expanded) * best_solution
        )

        if cumulative_cost is not None:
            cumulative_cost += best_cost / initial_cost

        # 6. Next Temperature & State
        next_temp = scheduler.step(step).to(device).repeat(current_solution.shape[0])
        current_temp = next_temp

        adv = torch.tensor(1 - (step / total_steps), device=device)
        model_temp = scale_to_unit(next_temp, config["STOP_TEMP"], config["INIT_TEMP"])
        if baseline:
            next_state = current_solution
        else:
            next_state = problem.to_state(
                *problem.build_state_components(current_solution, model_temp, adv)
            ).to(device)

        # 7. RL Reward (Only calculate if we are training/buffering)
        if replay_buffer is not None:
            reward_signal = calculate_reward(
                config,
                actual_improvement,
                is_valid,
                best_cost,
                old_best_cost,
                current_cost,
                cumulative_cost
                if cumulative_cost is not None
                else torch.zeros_like(best_cost),
                initial_cost,
                step,
                total_steps,
                epoch,
                step + 1 == total_steps,
            )

            replay_buffer.push(
                current_state,
                mask,
                action,
                next_state,
                reward_signal,
                action_log_prob,
                config["GAMMA"],
            )

        # Move to next state
        current_state = next_state.clone()

    # --- FINAL CLEANUP ---

    # Handle last transition in replay buffer if training
    if replay_buffer is not None and len(replay_buffer) > 0:
        replay_buffer.push(*(list(replay_buffer.pop()[:-1]) + [0.0]))

    # --- DUMMY RETURN ---
    # Only best_x and min_cost are real. Others are None or Dummy.
    results = {
        "best_x": best_solution,
        "min_cost": best_cost,
        # Dummies/Placeholders to prevent key errors in existing code
        "primal": torch.tensor(0.0),
        "ngain": torch.tensor(0.0),
        "n_acc": torch.tensor(0.0),
        "n_rej": torch.tensor(0.0),
        "distributions": None,
        "is_valid": None,
        "states": None,
        "actions": None,
        "acceptance": None,
        "costs": None,
        "init_cost": initial_cost,
        "reward": None,
        "average_sum_rewards": torch.tensor(0.0),
        "temperature": None,
        "best_step": torch.tensor(0.0),
        "capacity_left": None,
        "ratio": 0.0,
        "heuristic": None,
    }

    return results
