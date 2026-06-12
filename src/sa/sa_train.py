# ============================================================================
# SIMULATED ANNEALING TRAINER
# ============================================================================


import torch
from tensordict import TensorDict
from tqdm import tqdm

from model import SAModel
from problem import CVRP
from utils import capacity_utilization, extend_to

from .scheduler import Scheduler

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================


def scale_between(value: torch.Tensor, min_value: float, max_value: float) -> torch.Tensor:
    """Scales a [0, 1] value to [min, max]."""
    return min_value + (max_value - min_value) * value


def scale_to_unit(value: torch.Tensor, min_value: float, max_value: float) -> torch.Tensor:
    """Scales a [min, max] value to [0, 1]."""
    return (value - min_value) / (max_value - min_value)


def normalize(actual_improvement: torch.Tensor, initial_cost: torch.Tensor) -> torch.Tensor:
    """Scales improvement to [-1, 1] based on an expected max relative improvement."""
    MAX_EXPECTED_REL_IMPROVEMENT = 0.2
    relative_improvement = actual_improvement / initial_cost
    return torch.clamp(relative_improvement / MAX_EXPECTED_REL_IMPROVEMENT, -1.0, 1.0)


# ============================================================================
# INITIALIZATION HELPERS
# ============================================================================


def initialize_optimization_state(problem: CVRP, initial_solution: torch.Tensor, device: str):
    """Initializes a TensorDict for solution, costs, and tracking metrics."""
    best_cost = problem.cost(initial_solution)
    n = initial_solution.shape[0]

    return TensorDict(
        {
            "best_solution": initial_solution,
            "current_solution": initial_solution,
            "best_cost": best_cost,
            "current_cost": best_cost.clone(),
            "initial_cost": best_cost.clone(),
            "cumulative_cost": torch.ones_like(best_cost),
            "best_cost_step": torch.zeros_like(best_cost, dtype=torch.long),
            "capacity_left": capacity_utilization(
                initial_solution, problem.get_demands(initial_solution), problem.capacity
            ),
        },
        batch_size=[n],
        device=device,
    )


def initialize_tracking_variables():
    """Initializes history lists and counters."""
    return {
        "accepted_moves": 0,
        "rejected_moves": 0,
        "action_distributions": [],
        "solution_history": [],
        "state_history": [],
        "action_history": [],
        "is_valid_history": [],
        "temperature": [],
        "heuristic_choice": [],
        "acceptance_history": [],
        "cost_history": [],
        "all_rewards": [],
        "ratio": 0.0,
    }


# ============================================================================
# CORE LOGIC: ACCEPTANCE & REWARDS
# ============================================================================


def metropolis_accept(
    cost_improvement: torch.Tensor, current_temp: torch.Tensor, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Metropolis-Hastings acceptance criterion.
    Accept if improvement > 0 OR random < exp(improvement / temp).
    """
    # p = min(1, exp(gain/T))
    acceptance_prob = torch.minimum(
        torch.exp(cost_improvement / current_temp), torch.ones_like(cost_improvement)
    )
    random_sample = torch.rand(acceptance_prob.shape, device=device)
    is_accepted = (random_sample < acceptance_prob).long()

    # Filter improvement by acceptance
    actual_improvement = cost_improvement * is_accepted
    return is_accepted, actual_improvement


def calculate_reward(
    config: dict,
    actual_improvement: torch.Tensor,
    is_accepted: torch.Tensor,
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
    Computes RL reward signal based on the selected REWARD configuration.
    """
    if config["METHOD"] != "ppo":
        return torch.zeros_like(actual_improvement).view(-1, 1)

    reward = torch.zeros_like(actual_improvement).view(-1, 1)
    reward_mode = config["REWARD"]

    if reward_mode == "immediate":
        val_imm = (
            normalize(actual_improvement, initial_cost)
            if config["NORMALIZE_REWARD"]
            else actual_improvement
        )
        # Penalize invalid, zero for no-op, value for valid
        reward = torch.where(
            ~is_valid.bool().squeeze(-1),
            -1.5,
            torch.where(
                actual_improvement == 0,
                0.0,
                val_imm,
            ),
        ).view(-1, 1)

    elif reward_mode == "sa_aligned":
        val_imm = (
            normalize(actual_improvement, initial_cost)
            if config["NORMALIZE_REWARD"]
            else actual_improvement
        )
        is_acc = is_accepted.bool()
        if is_acc.dim() > 1:
            is_acc = is_acc.squeeze(-1)

        sa_reward = torch.where(
            is_acc & (actual_improvement > 0),
            val_imm,
            torch.where(
                is_acc & (actual_improvement <= 0),
                torch.zeros_like(val_imm),
                torch.full_like(val_imm, -0.1),  # Rejected move penalty
            ),
        )
        reward = torch.where(~is_valid.bool().squeeze(-1), -1.5, sa_reward).view(-1, 1)

    elif reward_mode == "global_best" or reward_mode == "dact":
        improvement_over_best = old_best_cost - best_cost
        val_imm = (
            normalize(improvement_over_best, initial_cost)
            if config["NORMALIZE_REWARD"]
            else improvement_over_best
        )
        reward = torch.where(
            improvement_over_best > 0, val_imm, torch.zeros_like(improvement_over_best)
        ).view(-1, 1)
        reward *= config.get("REWARD_SCALE", 1.0)

    elif reward_mode == "hybrid":
        # Combine immediate and global_best
        improvement_over_best = old_best_cost - best_cost
        val_global = (
            normalize(improvement_over_best, initial_cost)
            if config["NORMALIZE_REWARD"]
            else improvement_over_best
        )
        val_imm = (
            normalize(actual_improvement, initial_cost)
            if config["NORMALIZE_REWARD"]
            else actual_improvement
        )

        imm_reward = torch.where(
            ~is_valid.bool().squeeze(-1),
            -1.5,
            torch.where(actual_improvement == 0, 0.0, val_imm),
        ).view(-1, 1)

        global_reward = torch.where(
            improvement_over_best > 0, val_global, torch.zeros_like(val_global)
        ).view(-1, 1)

        alpha = config.get("HYBRID_ALPHA", 0.5)
        reward = alpha * imm_reward + (1 - alpha) * global_reward

    elif reward_mode == "curriculum":
        # Transition from sa_aligned to global_best based on epoch progress
        # total_epochs = config.get("N_EPOCHS", 1)
        total_epochs = 250
        progress = (epoch - 1) / max(1, total_epochs - 1)

        # Calculate sa_aligned part
        val_imm = (
            normalize(actual_improvement, initial_cost)
            if config["NORMALIZE_REWARD"]
            else actual_improvement
        )
        is_acc = is_accepted.bool().squeeze(-1) if is_accepted.dim() > 1 else is_accepted.bool()
        sa_reward = torch.where(
            is_acc & (actual_improvement > 0),
            val_imm,
            torch.where(
                is_acc & (actual_improvement <= 0),
                torch.zeros_like(val_imm),
                torch.full_like(val_imm, -0.1),
            ),
        )
        sa_part = torch.where(~is_valid.bool().squeeze(-1), -1.5, sa_reward).view(-1, 1)

        # Calculate global_best part
        improvement_over_best = old_best_cost - best_cost
        val_global = (
            normalize(improvement_over_best, initial_cost)
            if config["NORMALIZE_REWARD"]
            else improvement_over_best
        )
        global_part = torch.where(
            improvement_over_best > 0, val_global, torch.zeros_like(val_global)
        ).view(-1, 1)

        # Linear transition
        reward = (1 - progress) * sa_part + progress * global_part

    elif reward_mode == "curriculum_terminal":
        # Transition from sa_aligned to terminal based on epoch progress
        # total_epochs = config.get("N_EPOCHS", 1)
        total_epochs = 250
        progress = (epoch - 1) / max(1, total_epochs - 1)

        # Calculate sa_aligned part
        val_imm = (
            normalize(actual_improvement, initial_cost)
            if config["NORMALIZE_REWARD"]
            else actual_improvement
        )
        is_acc = is_accepted.bool().squeeze(-1) if is_accepted.dim() > 1 else is_accepted.bool()
        sa_reward = torch.where(
            is_acc & (actual_improvement > 0),
            val_imm,
            torch.where(
                is_acc & (actual_improvement <= 0),
                torch.zeros_like(val_imm),
                torch.full_like(val_imm, -0.1),
            ),
        )
        sa_part = torch.where(~is_valid.bool().squeeze(-1), -1.5, sa_reward).view(-1, 1)

        # Calculate terminal part
        terminal_part = torch.zeros_like(sa_part)
        if last_step:
            terminal_part = ((initial_cost - best_cost) / initial_cost).view(-1, 1)

        # Linear transition
        reward = (1 - progress) * sa_part + progress * terminal_part

    elif reward_mode == "step_penalty_init":
        reward = -(best_cost / initial_cost).view(-1, 1)

    elif reward_mode == "step_penalty_16":
        reward = -(best_cost / 16.0).view(-1, 1)

    elif reward_mode == "terminal":
        if last_step:
            reward = ((initial_cost - best_cost) / initial_cost).view(-1, 1)

    return reward


# ============================================================================
# MAIN TRAINING LOOP
# ============================================================================


def sa_train(
    actor: SAModel,
    problem: CVRP,
    initial_solution: torch.Tensor,
    config: dict,
    baseline: bool = False,
    random_std: float = 0.2,
    greedy: bool = False,
    record_state: bool = False,
    replay_buffer=None,
    train: bool = False,
    epoch: int = 0,
    device: str = "",
    desc_tqdm: str = "Simulated Annealing Progress",
    freeze_after: torch.Tensor = None,
) -> dict[str, torch.Tensor]:
    """
    Main Neural Simulated Annealing Loop.
    Generates actions via 'actor', accepts/rejects via Metropolis, and logs data.
    """
    if device == "":
        device = str(initial_solution.device)

    # Setup Scheduler
    total_steps = config["OUTER_STEPS"] if train else config["TEST_OUTER_STEPS"]

    scheduler = Scheduler(
        config["SCHEDULER"],
        T_max=config["INIT_TEMP"],
        T_min=config["STOP_TEMP"],
        step_max=total_steps,
    )

    # Initialize State & Tracking
    opt_state = initialize_optimization_state(problem, initial_solution, device)
    tracking = initialize_tracking_variables()
    if record_state:
        tracking["solution_history"].append(opt_state["current_solution"])

    current_temp = torch.tensor([1], device=device).repeat(opt_state["best_cost"].shape[0])
    current_temp = scale_between(current_temp, config["STOP_TEMP"], config["INIT_TEMP"])

    tracking["temperature"].append(current_temp.clone())
    tracking["cost_history"].append(opt_state["current_cost"].clone())

    # Build Initial State Tensor
    normalized_temp = scale_to_unit(current_temp, config["STOP_TEMP"], config["INIT_TEMP"])
    current_state = problem.to_state(
        *problem.build_state_components(
            opt_state["current_solution"],
            normalized_temp,
            torch.tensor(1.0, device=device),
        )
    ).to(device)

    # --- OPTIMIZATION LOOP ---

    progress_bar = tqdm(
        range(total_steps),
        desc=("Train/ " if train else "Test/ ") + desc_tqdm,
        colour="green",
        leave=False,
    )

    for step in progress_bar:
        # 1. Action Generation
        if record_state:
            tracking["state_history"].append(current_state)

        with torch.no_grad():
            if baseline:
                action, action_log_prob, mask, cond_rank = actor.baseline_sample(
                    current_state, problem=problem
                )
            else:
                action, action_log_prob, mask, cond_rank = actor.sample(
                    current_state, greedy=greedy, problem=problem
                )
        if "cuda" in device:
            torch.cuda.empty_cache()

        if record_state:
            tracking["action_distributions"].append(
                actor.get_logits(current_state, action, problem=problem)
            )
            tracking["action_history"].append(action)

        # 3. Apply Action & Evaluate
        sol_components, *_ = problem.from_state(current_state)
        proposed_sol, is_valid = problem.update(sol_components, action)
        proposed_cost = problem.cost(proposed_sol)

        cost_improvement = opt_state["current_cost"] - proposed_cost

        # 4. Metropolis Acceptance
        if config["METROPOLIS"]:
            is_accepted, actual_improvement = metropolis_accept(
                cost_improvement, current_temp, device
            )
        else:
            is_accepted = torch.ones_like(cost_improvement)
            actual_improvement = cost_improvement

        # Per-instance curriculum depth: once an instance reaches its sampled
        # warmup depth d_i, freeze it (reject every further move) so its returned
        # solution is the SA state at exactly d_i steps. No-op when freeze_after
        # is None (the default), so existing callers are unaffected.
        if freeze_after is not None:
            active = (step < freeze_after).to(is_accepted.dtype)
            is_accepted = is_accepted * active
            actual_improvement = actual_improvement * active

        tracking["is_valid_history"].append(is_valid.float().mean().item())
        tracking["accepted_moves"] += is_accepted
        tracking["rejected_moves"] += 1 - is_accepted
        if record_state:
            tracking["acceptance_history"].append(is_accepted)

        # 5. Update Current Solution

        # Update Cost: Select proposed_cost where accepted, else keep old cost

        opt_state["current_cost"] = (
            is_accepted * proposed_cost + (1 - is_accepted) * opt_state["current_cost"]
        )

        # Update Solution Tensor: Select proposed_sol where accepted, else keep old solution
        # Note: extend_to ensures the mask matches the solution shape [Batch, Nodes, 1]
        is_accepted_expanded = extend_to(is_accepted, sol_components)

        opt_state["current_solution"] = (
            is_accepted_expanded * proposed_sol + (1 - is_accepted_expanded) * sol_components
        ).long()

        # Sync problem internal state
        problem.update_tensor(opt_state["current_solution"])
        if record_state:
            tracking["solution_history"].append(opt_state["current_solution"])

        if record_state:
            tracking["cost_history"].append(opt_state["current_cost"])

        # 6. Update Best Solution
        old_best_cost = opt_state["best_cost"].clone()
        is_improvement = (opt_state["current_cost"] < opt_state["best_cost"]).long()

        is_imp_expanded = extend_to(is_improvement, opt_state["current_solution"])
        opt_state["best_solution"] = (
            is_imp_expanded * opt_state["current_solution"]
            + (1 - is_imp_expanded) * opt_state["best_solution"]
        )
        opt_state["best_cost"] = torch.minimum(opt_state["current_cost"], opt_state["best_cost"])
        opt_state["best_cost_step"] = torch.max(
            is_improvement * (step + 1), opt_state["best_cost_step"]
        )
        opt_state["cumulative_cost"] += opt_state["best_cost"] / opt_state["initial_cost"]

        # 7. Temperature & State Update
        next_temp = scheduler.step(step).to(device).repeat(opt_state["current_solution"].shape[0])
        current_temp = next_temp

        # Prepare next state
        adv = torch.tensor(1 - (step / total_steps), device=device)
        model_temp = scale_to_unit(next_temp, config["STOP_TEMP"], config["INIT_TEMP"])

        next_state = problem.to_state(
            *problem.build_state_components(opt_state["current_solution"], model_temp, adv)
        ).to(device)

        tracking["temperature"].append(current_temp.detach())

        # 8. RL Reward Calculation & Storage
        tracking["reward_signal"] = calculate_reward(
            config,
            actual_improvement,
            is_accepted,
            is_valid,
            opt_state["best_cost"],
            old_best_cost,
            opt_state["current_cost"],
            opt_state["cumulative_cost"],
            opt_state["initial_cost"],
            step,
            total_steps,
            epoch,
            step + 1 == total_steps,
        )
        tracking["all_rewards"].append(tracking["reward_signal"].flatten())

        if replay_buffer is not None:
            replay_buffer.push(
                current_state,
                mask,
                action,
                tracking["reward_signal"],
                action_log_prob,
                cond_rank=cond_rank,
            )

        # Move to next state
        current_state = next_state.clone()

    # --- FINALIZE RESULTS ---

    # Mark last transition as terminal (gamma=0)
    if replay_buffer is not None and len(replay_buffer) > 0:
        replay_buffer.mark_terminal()

    n_problems = opt_state["best_solution"].shape[0]

    results_td = TensorDict(
        {
            "best_x": opt_state["best_solution"],
            "min_cost": opt_state["best_cost"],
            "primal": opt_state["cumulative_cost"],
            "ngain": -(opt_state["initial_cost"] - opt_state["current_cost"]),
            "n_acc": tracking["accepted_moves"].float(),
            "n_rej": tracking["rejected_moves"].float(),
            "reward": tracking["reward_signal"],
            "best_step": opt_state["best_cost_step"].float(),
            "capacity_left": capacity_utilization(
                opt_state["best_solution"],
                problem.get_demands(opt_state["best_solution"]),
                problem.capacity,
            ),
            "init_cost": opt_state["initial_cost"],
        },
        batch_size=[n_problems],
    )

    results_extra = {
        "average_sum_rewards": torch.stack(tracking["all_rewards"]).cpu().sum(dim=0).mean(),
        "is_valid": torch.tensor(tracking["is_valid_history"]),
        "distributions": tracking["action_distributions"],
        "states": tracking["state_history"],
        "solutions": tracking["solution_history"],
        "actions": tracking["action_history"],
        "acceptance": tracking["acceptance_history"],
        "costs": tracking["cost_history"],
        "temperature": tracking["temperature"],
    }

    if len(config["HEURISTIC"]) > 1:
        results_extra["ratio"] = tracking["ratio"] / total_steps
        results_extra["heuristic"] = tracking["heuristic_choice"]

    return results_td, results_extra
