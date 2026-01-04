from typing import Dict, Tuple

import torch
from tqdm import tqdm

from model import SAModel
from problem import Problem
from utils import capacity_utilization, extend_to

from .scheduler import Scheduler

# ================================
# UTILITY FUNCTIONS
# ================================


def scale_between(value: float, min_value: float, max_value: float) -> float:
    """Scale a value in [0, 1] to the range [min_value, max_value]."""
    return min_value + (max_value - min_value) * value


def scale_to_unit(value: float, min_value: float, max_value: float) -> float:
    """Scale a value in [min_value, max_value] to the range [0, 1]."""
    return (value - min_value) / (max_value - min_value)


def normalize(actual_improvement, initial_cost) -> torch.Tensor:
    """
    Scale the improvement to a reward signal in [-1, 1] based on expected max
    improvement.
    Args:
        actual_improvement: The actual improvement achieved
        initial_cost: The initial cost before improvement
    Returns:
        Scaled and clamped reward tensor
    """
    MAX_EXPECTED_REL_IMPROVEMENT = 0.2

    # Calculate relative improvement
    relative_improvement = actual_improvement / initial_cost

    # Scale and clamp with your fixed limit
    scaled_reward = relative_improvement / MAX_EXPECTED_REL_IMPROVEMENT
    clamped_reward = torch.clamp(scaled_reward, -1.0, 1.0)
    return clamped_reward


# ================================
# ACCEPTANCE CRITERIA
# ================================


def p_accept(gain: torch.Tensor, temp: torch.Tensor) -> torch.Tensor:
    """
    Compute the Metropolis acceptance probability for a proposed move.

    Args:
        gain: Energy difference (current_cost - proposed_cost)
        temp: Current temperature

    Returns:
        Probability to accept the move, following exp(gain/temp) with clipping to [0,1]
    """
    return torch.minimum(torch.exp(gain / temp), torch.ones_like(gain))


def metropolis_accept(
    cost_improvement: torch.Tensor, current_temp: torch.Tensor, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply the Metropolis acceptance criterion.

    Args:
        cost_improvement: The difference between current and proposed cost
        current_temp: Current temperature
        device: Device for calculations

    Returns:
        Tuple of (is_accepted, actual_improvement)
    """
    acceptance_prob = p_accept(cost_improvement, current_temp)
    random_sample = torch.rand(acceptance_prob.shape, device=device)
    is_accepted = (random_sample < acceptance_prob).long()
    actual_improvement = cost_improvement * is_accepted
    return is_accepted, actual_improvement


# ================================
# INITIALIZATION HELPERS
# ================================


def initialize_optimization_state(
    problem: Problem, initial_solution: torch.Tensor, device: torch.device
):
    """Initialize all optimization tracking variables."""
    best_solution = current_solution = initial_solution
    best_cost = current_cost = problem.cost(best_solution)
    capacity_left = capacity_utilization(
        best_solution, problem.get_demands(best_solution), problem.capacity
    )
    initial_cost = best_cost.clone()
    cumulative_cost = best_cost.clone() / initial_cost
    best_cost_step = torch.zeros_like(best_cost, dtype=torch.long)

    return {
        "best_solution": best_solution,
        "current_solution": current_solution,
        "best_cost": best_cost,
        "current_cost": current_cost,
        "capacity_left": capacity_left,
        "initial_cost": initial_cost,
        "cumulative_cost": cumulative_cost,
        "best_cost_step": best_cost_step,
    }


def initialize_temperature(config: dict, batch_size: int, device: torch.device):
    """Initialize temperature schedule."""
    current_temp = torch.tensor([1], device=device).repeat(batch_size)
    current_temp = scale_between(current_temp, config["STOP_TEMP"], config["INIT_TEMP"])
    return current_temp


def initialize_tracking_variables():
    """Initialize all tracking and history variables."""
    return {
        "accepted_moves": 0,
        "rejected_moves": 0,
        "action_distributions": [],
        "state_history": [],
        "action_history": [],
        "is_valid_history": [],
        "temperature": [],
        "heuristic_choice": [],
        "acceptance_history": [],
        "cost_history": [],
        "reward_signal": None,
        "ratio": 0.0,
    }


# ================================
# ACTION GENERATION AND PROCESSING
# ================================


def generate_action(
    actor: SAModel,
    current_state: torch.Tensor,
    baseline: bool,
    greedy: bool,
    random_std: float,
    train: bool,
    problem: Problem,
    device: torch.device,
):
    """Generate action from policy or baseline."""
    with torch.no_grad():
        with torch.amp.autocast(device.type):
            if baseline:
                res = actor.baseline_sample(current_state, problem=problem)
            else:
                res = actor.sample(current_state, greedy=greedy, problem=problem)
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return res


def process_heuristic_action(action: torch.Tensor, config: dict, tracking: dict):
    """Process heuristic action tracking if using mixed approach."""
    if config["HEURISTIC"] == "mix":
        heuristic_action = sum(action[:, 2])
        tracking["ratio"] += heuristic_action / len(action[:, 2])
        tracking["heuristic_choice"].append(heuristic_action)


def record_action_info(
    actor: SAModel,
    current_state: torch.Tensor,
    action: torch.Tensor,
    problem: Problem,
    tracking: dict,
    record_state: bool,
):
    """Record action information for analysis."""
    if record_state:
        tracking["action_distributions"].append(
            actor.get_logits(current_state, action, problem=problem)
        )
        tracking["action_history"].append(action)


# ================================
# SOLUTION UPDATE AND ACCEPTANCE
# ================================


def evaluate_and_accept_solution(
    problem: Problem,
    current_state: torch.Tensor,
    action: torch.Tensor,
    current_cost: torch.Tensor,
    current_temp: torch.Tensor,
    config: dict,
    device: torch.device,
):
    """Evaluate proposed solution and determine acceptance."""
    # Generate and evaluate proposed solution
    solution_components, *_ = problem.from_state(current_state)
    proposed_solution, is_valid = problem.update(solution_components, action)
    proposed_cost = problem.cost(proposed_solution)

    # Calculate cost improvement
    cost_improvement = current_cost - proposed_cost

    # Apply acceptance criterion
    if config["METROPOLIS"]:
        is_accepted, actual_improvement = metropolis_accept(
            cost_improvement, current_temp, device
        )
    else:
        is_accepted = torch.ones_like(cost_improvement)
        actual_improvement = cost_improvement

    return {
        "solution_components": solution_components,
        "proposed_solution": proposed_solution,
        "proposed_cost": proposed_cost,
        "is_valid": is_valid,
        "is_accepted": is_accepted,
        "actual_improvement": actual_improvement,
    }


def update_current_solution(
    solution_data: dict, current_cost: torch.Tensor, current_solution: torch.Tensor
):
    """Update current solution based on acceptance."""
    is_accepted = solution_data["is_accepted"]
    proposed_cost = solution_data["proposed_cost"]
    proposed_solution = solution_data["proposed_solution"]
    solution_components = solution_data["solution_components"]

    # Update current cost and solution
    new_current_cost = is_accepted * proposed_cost + (1 - is_accepted) * current_cost
    is_accepted_expanded = extend_to(is_accepted, solution_components)
    new_current_solution = (
        is_accepted_expanded * proposed_solution
        + (1 - is_accepted_expanded) * solution_components
    ).long()

    return new_current_cost, new_current_solution


def update_best_solution(
    current_cost: torch.Tensor,
    current_solution: torch.Tensor,
    initial_cost: torch.Tensor,
    best_cost: torch.Tensor,
    best_solution: torch.Tensor,
    step: int,
    best_cost_step: torch.Tensor,
    cumulative_cost: torch.Tensor,
):
    """Update best solution tracking."""
    is_improvement = (current_cost < best_cost).long()
    is_improvement_expanded = extend_to(is_improvement, current_solution)

    new_best_solution = (
        is_improvement_expanded * current_solution
        + (1 - is_improvement_expanded) * best_solution
    )
    new_best_cost_step = torch.max(is_improvement * (step + 1), best_cost_step)
    new_best_cost = torch.minimum(current_cost, best_cost)
    new_cumulative_cost = cumulative_cost + (new_best_cost / initial_cost)

    return new_best_solution, new_best_cost, new_best_cost_step, new_cumulative_cost


# ================================
# CAPACITY AND REWARD PROCESSING
# ================================


def calculate_reward(
    config: dict,
    actual_improvement: torch.Tensor,
    is_valid: torch.Tensor,
    best_cost: torch.Tensor,
    cumulative_cost: torch.Tensor,
    initial_cost: torch.Tensor,
    last_step: bool = False,
):
    """Calculate reward signal for reinforcement learning."""
    if config["METHOD"] != "ppo":
        return None

    reward_signal = torch.zeros_like(actual_improvement).view(-1, 1)

    # 1. Calculate Base Reward Strategy
    if config["REWARD"] == "immediate":
        # Determine value based on improvement
        if config["NORMALIZE_REWARD"]:
            value = normalize(actual_improvement, initial_cost)
        else:
            value = actual_improvement

        # Apply immediate-specific logic:
        # - Invalid solutions get -1.5
        # - Zero improvement gets 0.0
        # - Otherwise use calculated value
        reward_signal = torch.where(
            ~is_valid.bool().squeeze(-1),
            -1.5,
            torch.where(
                actual_improvement == 0,
                0.0,
                value,
            ),
        ).view(-1, 1)

    elif config["REWARD"] == "min_cost":
        reward_signal = ((initial_cost + best_cost) / initial_cost).view(-1, 1)

    elif config["REWARD"] == "primal":
        reward_signal = -cumulative_cost.view(-1, 1)

    # 2. Apply Global Invalid Penalty (overrides specific strategy penalties)
    if config["REWARD_VALID"]:
        reward_signal[~is_valid.view(-1, 1)] = -1.0

    # 3. Apply Last Step Bonus (overrides previous rewards)
    if config["REWARD_LAST"] and last_step:
        reward_signal = config["REWARD_LAST_SCALE"] * (
            (initial_cost - best_cost) / initial_cost
        ).view(-1, 1)

    return reward_signal


# ================================
# TEMPERATURE AND STATE UPDATES
# ================================


def update_temperature_and_state(
    step: int,
    config: dict,
    scheduler: Scheduler,
    current_solution: torch.Tensor,
    problem: Problem,
    device: torch.device,
):
    """Update temperature and prepare next state."""
    # Calculate advancement ratio (decreases from 1 to 0)
    adv = torch.tensor(1 - (step / config["OUTER_STEPS"]), device=device)

    # Use predefined cooling schedule
    next_temp = scheduler.step(step).to(device).repeat(current_solution.shape[0])

    # Normalize temperature for state representation
    model_next_temp = scale_to_unit(next_temp, config["STOP_TEMP"], config["INIT_TEMP"])

    # Update problem's internal tensor representations
    problem.update_tensor(current_solution)

    # Build next state
    next_state = problem.to_state(
        *problem.build_state_components(current_solution, model_next_temp, adv)
    ).to(device)

    return next_temp, next_state


# ================================
# MAIN TRAINING FUNCTION
# ================================


def sa_train(
    actor: SAModel,
    problem: Problem,
    initial_solution: torch.Tensor,
    config: dict,
    baseline: bool = False,
    random_std: float = 0.2,
    greedy: bool = False,
    record_state: bool = False,
    replay_buffer=None,
    train: bool = False,
    device: str = None,
    desc_tqdm: str = "Simulated Annealing Progress",
) -> Dict[str, torch.Tensor]:
    """
    Perform Simulated Annealing optimization for combinatorial problems.

    This function implements a neural-guided simulated annealing algorithm that uses
    a policy network to propose solution modifications while following a temperature
    cooling schedule for acceptance decisions.
    """
    # Device setup
    if device is None:
        device = initial_solution.device

    # Initialize scheduler
    scheduler = Scheduler(
        config["SCHEDULER"],
        T_max=config["INIT_TEMP"],
        T_min=config["STOP_TEMP"],
        step_max=config["OUTER_STEPS"],
    )

    # Initialize all optimization state variables
    opt_state = initialize_optimization_state(problem, initial_solution, device)
    current_temp = initialize_temperature(
        config, opt_state["best_cost"].shape[0], device
    )
    tracking = initialize_tracking_variables()

    # Add initial temperature to tracking
    tracking["temperature"].append(current_temp.clone())
    tracking["cost_history"].append(opt_state["current_cost"].clone())

    # Prepare initial state representation
    normalized_temp = scale_to_unit(
        current_temp, config["STOP_TEMP"], config["INIT_TEMP"]
    )
    components = problem.build_state_components(
        opt_state["current_solution"],
        normalized_temp,
        torch.tensor(1.0, device=device),  # Initial progress
    )
    current_state = problem.to_state(*components).to(device)

    # Progress bar setup
    desc = ("Train/ " if train else "Test/ ") + desc_tqdm

    step = config["OUTER_STEPS"] if train else config["TEST_OUTER_STEPS"]
    # Main optimization loop
    for step in tqdm(
        range(step),
        desc=desc,
        colour="green",
        unit="step",
        leave=False,
    ):
        # Inner loop at fixed temperature
        for inner_step in range(config["INNER_STEPS"]):
            # Record current state if needed
            if record_state:
                tracking["state_history"].append(current_state)

            # Generate action from policy
            action, action_log_prob, mask = generate_action(
                actor,
                current_state,
                baseline,
                greedy,
                random_std,
                train,
                problem,
                device,
            )

            # Process heuristic tracking
            process_heuristic_action(action, config, tracking)

            # Record action information
            record_action_info(
                actor, current_state, action, problem, tracking, record_state
            )

            # Evaluate proposed solution and determine acceptance
            solution_data = evaluate_and_accept_solution(
                problem,
                current_state,
                action,
                opt_state["current_cost"],
                current_temp,
                config,
                device,
            )

            # Track validity metrics
            tracking["is_valid_history"].append(
                solution_data["is_valid"].to(torch.float32).mean().item()
            )

            # Update move counters
            tracking["accepted_moves"] += solution_data["is_accepted"]
            tracking["rejected_moves"] += 1 - solution_data["is_accepted"]

            # Record acceptance if needed
            if record_state:
                tracking["acceptance_history"].append(solution_data["is_accepted"])

            # Update current solution
            opt_state["current_cost"], opt_state["current_solution"] = (
                update_current_solution(
                    solution_data,
                    opt_state["current_cost"],
                    opt_state["current_solution"],
                )
            )

            # Record cost history
            if record_state:
                tracking["cost_history"].append(opt_state["current_cost"])

            # Update best solution tracking
            (
                opt_state["best_solution"],
                opt_state["best_cost"],
                opt_state["best_cost_step"],
                opt_state["cumulative_cost"],
            ) = update_best_solution(
                opt_state["current_cost"],
                opt_state["current_solution"],
                opt_state["initial_cost"],
                opt_state["best_cost"],
                opt_state["best_solution"],
                step,
                opt_state["best_cost_step"],
                opt_state["cumulative_cost"],
            )

            # Update temperature and state at end of inner steps
            if inner_step == config["INNER_STEPS"] - 1:
                current_temp, next_state = update_temperature_and_state(
                    step,
                    config,
                    scheduler,
                    opt_state["current_solution"],
                    problem,
                    device,
                )
                tracking["temperature"].append(current_temp.detach())

            # Calculate reward for RL
            tracking["reward_signal"] = calculate_reward(
                config,
                solution_data["actual_improvement"],
                solution_data["is_valid"],
                opt_state["best_cost"],
                opt_state["cumulative_cost"],
                opt_state["initial_cost"],
                step + 1 == config["OUTER_STEPS"],
            )

            # Store experience in replay buffer
            if replay_buffer is not None:
                replay_buffer.push(
                    current_state,
                    mask,
                    action,
                    next_state,
                    tracking["reward_signal"],
                    action_log_prob,
                    config["GAMMA"],
                )

        # Update state and temperature for next iteration
        current_state = next_state.clone()

    # Finalize results
    negative_improvement = -(opt_state["initial_cost"] - opt_state["current_cost"])

    # Handle final transition in replay buffer
    if replay_buffer is not None and len(replay_buffer) > 0:
        final_transition = replay_buffer.pop()
        replay_buffer.push(*(list(final_transition[:-1]) + [0.0]))

    final_capacity_left = capacity_utilization(
        opt_state["best_solution"],
        problem.get_demands(opt_state["best_solution"]),
        problem.capacity,
    )

    # Build results dictionary
    results = {
        "best_x": opt_state["best_solution"],
        "min_cost": opt_state["best_cost"],
        "primal": opt_state["cumulative_cost"],
        "ngain": negative_improvement,
        "n_acc": tracking["accepted_moves"].float(),
        "n_rej": tracking["rejected_moves"].float(),
        "distributions": tracking["action_distributions"],
        "is_valid": torch.tensor(tracking["is_valid_history"]),
        "states": tracking["state_history"],
        "actions": tracking["action_history"],
        "acceptance": tracking["acceptance_history"],
        "costs": tracking["cost_history"],
        "init_cost": opt_state["initial_cost"],
        "reward": tracking["reward_signal"],
        "temperature": tracking["temperature"],
        "best_step": opt_state["best_cost_step"].float(),
        "capacity_left": final_capacity_left,
    }

    # Add heuristic-specific results
    if len(config["HEURISTIC"]) > 1:
        results["ratio"] = tracking["ratio"] / config["OUTER_STEPS"]
        results["heuristic"] = tracking["heuristic_choice"]

    return results
