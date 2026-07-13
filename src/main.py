"""
CVRP Solver: PPO + Simulated Annealing
======================================
This module implements a hybrid Reinforcement Learning approach for the
Capacitated Vehicle Routing Problem (CVRP).

Key Components:
1. Actor-Critic Policy (PPO)
2. Simulated Annealing (SA) for refinement
3. Curriculum Learning for progressive difficulty
"""

# ============================================================================
# IMPORTS
# ============================================================================

# Standard Library
import math
import time
import warnings
from typing import Any

# Third-Party Libraries
import numpy as np
import torch
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import tqdm

# Local Modules
from algo import P_generate_instances, stack_res
from init import init_problem, initialize_models, initialize_test_problem, test_model
from model import SAModel
from ppo import ReplayBuffer, ppo
from problem import CVRP
from sa import sa_train
from setup import _HP, WandbLogger, get_script_arguments
from utils import setup_device, setup_logging, setup_reproducibility

# ============================================================================
# CONFIGURATION & SETUP
# ============================================================================

# Suppress specific library warnings
warnings.filterwarnings("ignore", message="Attempting to run cuBLAS")

# Initialize Logging
logger = setup_logging()

# Load Configuration
config = _HP("src/HyperParameters/HP.yaml")
config.update(get_script_arguments(config.keys()))

# Initialize Experiment Tracking (WandB)
if config["LOG"]:
    WandbLogger.init(None, 3, config)
    logger.info(f"WandB model save directory: {WandbLogger.get_model_dir()}")


# ============================================================================
# LOGGING & UTILITIES
# ============================================================================


def log_training_and_test_metrics(
    actor_loss: float | None,
    critic_loss: float,
    steps: int,
    avg_actor_grad: float,
    avg_critic_grad: float,
    lr_actor: float,
    beta_kl: float,
    explained_var: float,
    rewards_mean: float,
    average_kl: float,
    entropy: float,
    pre_step: int,
    early_stopping_counter: int,
    a_min_cost: float,
    test_results_td: Any | None,
    test_results_extra: dict | None,
    epoch: int,
    config: dict[str, Any],
    log_test: bool = False,
) -> None:
    """Logs training and testing metrics to WandB."""
    logs = {}

    # 1. Training Metrics
    if actor_loss is not None:
        logs.update(
            {
                "Actor_loss": actor_loss,
                "Critic_loss": critic_loss,
                "Train_Steps": steps,
                "Train_loss": actor_loss + 0.5 * critic_loss,
                "Avg_actor_grad": avg_actor_grad,
                "Avg_critic_grad": avg_critic_grad,
                "LR_actor": lr_actor,
                "Beta_KL": beta_kl,
                "Explained_Variance": explained_var,
                "Average_KL": average_kl,
                "Entropy": entropy,
                "early_stopping_counter": early_stopping_counter,
                "Pre_step": pre_step,
                "Rewards_mean": rewards_mean,
            }
        )

    # 2. Test Metrics (Periodic)
    if log_test and test_results_td is not None and test_results_extra is not None:
        logs.update(
            {
                "Min_cost": torch.mean(test_results_td["min_cost"]),
                "A_min_cost": a_min_cost,
                "Gain": torch.mean(test_results_td["init_cost"] - test_results_td["min_cost"]),
                "Test_Rewards_mean": test_results_extra["average_sum_rewards"].item(),
                "Acceptance_rate": torch.mean(test_results_td["n_acc"])
                / config["TEST_OUTER_STEPS"],
                "Step_best_cost": torch.mean(test_results_td["best_step"])
                / config["TEST_OUTER_STEPS"],
                "Valid_percentage": torch.mean(test_results_extra["is_valid"]),
                "Final_capacity_left": torch.mean(test_results_td["capacity_left"]),
            }
        )

    WandbLogger.log(logs)


def save_model(path: str, actor_model: torch.nn.Module | None = None) -> None:
    """Saves the actor model state dictionary."""
    if actor_model is None:
        raise ValueError("No model provided for saving")

    torch.save(actor_model.state_dict(), path)
    if config["VERBOSE"]:
        logger.info(f"Model saved to: {path}")


def calculate_curriculum_steps_sig(step: int, config: dict[str, Any]) -> int:
    """
    Calculates the number of deterministic improvement steps (T_init)
    based on a Sigmoid schedule.
    """
    xi_cl = config["MAX_OUTER_STEPS_CL"]  # Max curriculum steps
    E = config["MAX_PROB_STEP"]  # Total curriculum epochs

    # Scale kappa based on total steps to maintain curve shape relative to paper
    # Paper used kappa=0.2 for 200 epochs
    kappa_base = config["STEEP_SIG"]  # Base steepness from config
    epochs_paper = 200
    kappa = kappa_base * (epochs_paper / max(1, E))

    def sigmoid_schedule(x, total_steps, k):
        exponent = -k * (x - total_steps / 2.0)
        exponent = max(-20.0, min(20.0, exponent))  # Clamp for stability
        return 1.0 / (1.0 + math.exp(exponent))

    s_0 = sigmoid_schedule(0, E, kappa)
    s_E = sigmoid_schedule(E, E, kappa)
    s_e = sigmoid_schedule(step, E, kappa)

    # Calculate progress ratio
    progress_ratio = (s_e - s_0) / (s_E - s_0) if E > 0 else 1.0

    t_init = int(progress_ratio * xi_cl)
    return max(0, min(t_init, xi_cl))


def calculate_curriculum_steps_lin(step: int, config: dict[str, Any]) -> int:
    xi_cl = config["MAX_OUTER_STEPS_CL"]
    E = config["MAX_PROB_STEP"]

    # LINEAR SCHEDULE: smoothly goes from 0 to xi_cl over E epochs
    progress_ratio = step / max(1, E)

    t_init = int(progress_ratio * xi_cl)
    return max(0, min(t_init, xi_cl))


def initialize_training_problem(
    problem: CVRP, device: str, config: dict[str, Any], epoch: int
) -> CVRP:
    """Regenerates the problem instance for the next training epoch."""
    if config["DATA"] == "uchoa":
        # Load structured instances
        epoch_seed = config["SEED"] + epoch
        coords_list, demands_list, capacity_list, _ = P_generate_instances(
            config["N_PROBLEMS"], epoch_seed, config["PROBLEM_DIM"]
        )
        coords, demands, capacity = stack_res(coords_list, demands_list, capacity_list)
        problem.generate_params(coords, demands.to(torch.int64), capacity)

    elif config["DATA"] == "random":
        # Generate fully random instances
        coords = torch.rand(config["N_PROBLEMS"], config["PROBLEM_DIM"] + 1, 2, device=device)
        demands = torch.randint(
            1, 10, (config["N_PROBLEMS"], config["PROBLEM_DIM"] + 1), device=device
        )
        demands[:, 0] = 0  # Depot has no demand
        capacity = torch.full((config["N_PROBLEMS"], 1), config["MAX_LOAD"], device=device)
        problem.generate_params(coords, demands, capacity)

    return problem


def refresh_chain_pool(
    problem: CVRP,
    pool: dict[str, torch.Tensor] | None,
    device: str,
    config: dict[str, Any],
    epoch: int,
) -> tuple[CVRP, dict[str, torch.Tensor], torch.Tensor]:
    """Persistent-chain curriculum (CL_MODE="chain").

    Instead of re-warming fresh instances every epoch (CL_MODE="warmup"),
    instances and their best solutions persist across epochs and the search
    resumes where it left off. Each epoch a fraction CHAIN_REFRESH of the
    chains is replaced by a fresh instance restarted from a raw construction,
    so chain ages follow a geometric distribution: a single batch mixes raw
    starts with solutions refined over many epochs (mean resume depth =
    OUTER_STEPS / CHAIN_REFRESH SA steps), at zero warmup compute.
    """
    n = config["N_PROBLEMS"]

    # Fresh instances for the whole batch; kept rows are overwritten below.
    problem = initialize_training_problem(problem, device, config, epoch)

    keep = None
    if pool is not None:
        keep = torch.rand(n, device=device) >= config["CHAIN_REFRESH"]
        max_age = config.get("CHAIN_MAX_AGE", 0)
        if max_age:
            keep &= pool["age"] < max_age
        coords = torch.where(keep.view(-1, 1, 1), pool["coords"], problem.coords)
        demands = torch.where(keep.view(-1, 1), pool["demands"], problem.demands)
        capacity = torch.where(keep.view(-1, 1), pool["capacity"], problem.capacity)
        problem.generate_params(coords, demands, capacity)

    # Raw constructions for every row (cheap); kept rows resume their stored
    # best solution instead. Widths may differ across epochs (route-count
    # slots), so both tensors are padded with depot indices (empty routes).
    solutions = problem.generate_init_state(
        init_heuristic=config["INIT"],
        multi_init=config["MULTI_INIT"],
        init_list=config.get("INIT_LIST", []),
    )
    if pool is not None:
        stored = pool["solutions"]
        width = max(solutions.shape[1], stored.shape[1])
        solutions = torch.nn.functional.pad(solutions, (0, 0, 0, width - solutions.shape[1]))
        stored = torch.nn.functional.pad(stored, (0, 0, 0, width - stored.shape[1]))
        solutions = torch.where(keep.view(-1, 1, 1), stored, solutions)
        solutions = problem.init_parameters(solutions)
        age = torch.where(keep, pool["age"] + 1, torch.zeros_like(pool["age"]))
    else:
        age = torch.zeros(n, dtype=torch.long, device=device)

    if config["VERBOSE"]:
        logger.info(
            f"[chain] epoch {epoch}: mean age {age.float().mean().item():.1f} epochs, "
            f"refreshed {n - int(keep.sum().item()) if keep is not None else n}/{n}, "
            f"mean resume cost {problem.cost(solutions).mean().item():.3f}"
        )

    pool = {
        "coords": problem.coords,
        "demands": problem.demands,
        "capacity": problem.capacity,
        "solutions": solutions,
        "age": age,
    }
    return problem, pool, solutions


# ============================================================================
# CORE TRAINING LOGIC
# ============================================================================


def train_ppo(
    actor: SAModel,
    critic: torch.nn.Module,
    actor_optimizer: torch.optim.Optimizer,
    critic_optimizer: torch.optim.Optimizer,
    critic_scheduler: ExponentialLR,
    problem: CVRP,
    config: dict[str, Any],
    step: int = 0,
    initial_solutions_override: torch.Tensor | None = None,
) -> tuple[dict, tuple, float, float, int]:
    """
    Executes a single training epoch (SA collection + PPO update).

    initial_solutions_override: resumed solutions from the persistent-chain
    curriculum (CL_MODE="chain"); bypasses construction and the warmup phase.
    """
    if problem.device == "cuda":
        torch.cuda.empty_cache()

    # -------------------------------------------------------
    # 1. Dynamic Initialization (Curriculum)
    # -------------------------------------------------------

    # Default to empty or specific list
    current_init_list = config.get("INIT_LIST", [])

    if config["MULTI_INIT"]:
        total_methods = len(current_init_list)
        target_step = config.get("MULTI_INIT_STEP", 0)

        if target_step > 0:
            # Calculate progress from 0.0 to 1.0
            # We cap progress at 1.0 so we don't go out of bounds after step 300
            progress = min(1.0, step / target_step)

            # Map progress to the number of methods [1 to total_methods]
            # Logic: Always use at least 1. Linearly add others.
            # Step 0   -> 1 method
            # Step 150 -> Half of methods
            # Step 300 -> All methods
            num_active = 1 + int(progress * (total_methods - 1))

            # Slice the list to get currently active methods
            current_init_list = current_init_list[:num_active]

        # (Optional) Log just to see it working
        # print(f"Step {step}: Using {len(current_init_list)} methods: {current_init_list}")

    # Generate initial solutions with the dynamic list, unless the persistent-
    # chain curriculum already provides resumed solutions for this epoch
    if initial_solutions_override is not None:
        initial_solutions = initial_solutions_override
    else:
        initial_solutions = problem.generate_init_state(
            init_heuristic=config["INIT"],
            multi_init=config["MULTI_INIT"],
            init_list=current_init_list,  # Pass the dynamic list here
        )
    buffer_size = config["OUTER_STEPS"]
    replay_buffer = ReplayBuffer(buffer_size, device=torch.device(problem.device))
    pre_step = 0

    # 2. Curriculum Learning (Improvement Phase) — warmup mechanism only;
    #    CL_MODE="chain" is handled upstream via initial_solutions_override

    if config["CL"] and config.get("CL_MODE", "warmup") == "warmup":
        t_init = (
            calculate_curriculum_steps_sig(step, config)
            if config["CL_TYPE"] == "sig"
            else calculate_curriculum_steps_lin(step, config)
        )

        if t_init > 0:
            original_steps = config.get("TEST_OUTER_STEPS", 0)
            config["TEST_OUTER_STEPS"] = t_init
            pre_step = t_init

            # Per-instance random depth: instead of warming every instance to the
            # same depth t_init, draw d_i ~ U{0..t_init} per instance so each batch
            # covers the full cost spectrum (d_i=0 -> raw init, d_i=t_init -> full
            # warmup). Same batch compute as the uniform warmup (frozen instances
            # still run the loop, their moves are just masked out).
            freeze_after = None
            if config.get("CL_RANDOM_DEPTH", False):
                n_inst = initial_solutions.shape[0]
                freeze_after = torch.randint(
                    0, t_init + 1, (n_inst,), device=initial_solutions.device
                )

            # Warmup: a compressed SA run over t_init steps (stochastic
            # sampling + Metropolis, NOT greedy descent); the best solution
            # found becomes the collection start point. No transitions are
            # stored (replay_buffer=None), so warmup compute is not trained on.
            pre_res_td, _ = sa_train(
                actor=actor,
                problem=problem,
                initial_solution=initial_solutions,
                config=config,
                replay_buffer=None,
                baseline=False,
                greedy=False,
                train=False,
                freeze_after=freeze_after,
            )

            # Use improved solutions as start point for training
            config["TEST_OUTER_STEPS"] = original_steps
            initial_solutions = pre_res_td["best_x"].detach()
            problem.init_parameters(initial_solutions)
            del pre_res_td

    # 3. Experience Collection (Simulated Annealing)
    sa_results_td, sa_results_extra = sa_train(
        actor=actor,
        problem=problem,
        initial_solution=initial_solutions,
        config=config,
        replay_buffer=replay_buffer,
        epoch=step,
        baseline=False,
        greedy=False,
        train=True,
    )

    # 4. Policy Optimization (PPO)
    train_stats = ppo(
        actor=actor,
        critic=critic,
        pb_dim=initial_solutions.shape[1],
        replay=replay_buffer,
        actor_opt=actor_optimizer,
        critic_opt=critic_optimizer,
        curr_epoch=step,
        cfg=config,
    )

    # 5. Scheduling & Monitoring
    critic_scheduler.step()

    def get_avg_grad(model):
        grads = [p.grad.abs().mean().item() for p in model.parameters() if p.grad is not None]
        return float(np.mean(grads)) if grads else 0.0

    avg_actor_grad = get_avg_grad(actor)
    avg_critic_grad = get_avg_grad(critic)

    if problem.device == "cuda":
        torch.cuda.empty_cache()

    return sa_results_td, sa_results_extra, train_stats, avg_actor_grad, avg_critic_grad, pre_step


# ============================================================================
# BENCHMARK HELPERS
# ============================================================================


def _log_benchmark_epoch(
    path: str,
    epoch: int,
    elapsed: float,
    actor_loss: float,
    critic_loss: float,
    test_loss: float,
) -> None:
    import csv
    import os

    write_header = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["epoch", "time_s", "actor_loss", "critic_loss", "test_loss"])
        w.writerow(
            [epoch, f"{elapsed:.3f}", f"{actor_loss:.6f}", f"{critic_loss:.6f}", f"{test_loss:.6f}"]
        )


# ============================================================================
# MAIN EXECUTION LOOP
# ============================================================================


def main(config: dict) -> None:
    """Main Orchestrator for CVRP Training."""

    # --- 1. Environment Setup ---
    device = setup_device(config["DEVICE"])
    config["DEVICE"] = device
    logger.info(
        f"Device: {device} | CUDA: {torch.cuda.get_device_name(0) if device == 'cuda' else 'N/A'}"
    )

    setup_reproducibility(config["SEED"], train=True)
    logger.info(f"Random Seed: {config['SEED']}")

    training_problem, input_dim = init_problem(
        config, dim=config["PROBLEM_DIM"], n_problem=config["N_PROBLEMS"]
    )
    config["ENTRY"] = input_dim
    logger.info(f"Problem Input Dimension: {input_dim}")

    # --- 2. Test Environment Setup ---
    test_problem, initial_test_solutions = initialize_test_problem(
        config,
        config["TEST_DIMENSION"],
        config["TEST_NB_PROBLEMS"],
        config["TEST_INIT"],
        "nazari" if config["NAZARI"] else "uchoa",
        device,
    )
    init_test_cost = torch.mean(test_problem.cost(initial_test_solutions))
    logger.info(f"Test Env Initialized | Cost: {init_test_cost.item():.2f}")

    # --- 3. Model & Optimizer Setup ---
    actor, critic = initialize_models(
        config["MODEL"],
        config["CRITIC_MODEL"],
        config["EMBEDDING_DIM"],
        config["ENTRY"],
        config["NUM_H_LAYERS"],
        config["UPDATE_METHOD"],
        config["HEURISTIC"],
        config["SEED"],
        device=device,
        cond_rank=config.get("COND_RANK", False),
        cond_detour=config.get("COND_DETOUR", False),
        global_context=config.get("GLOBAL_CONTEXT", False),
        bilinear=config.get("BILINEAR", False),
        logit_clip=config.get("LOGIT_CLIP", 0.0),
        learnable_temp=config.get("LEARNABLE_TEMP", False),
    )
    logger.info("Models Initialized")

    actor_optimizer = torch.optim.Adam(
        actor.parameters(), lr=config["LR_ACTOR"], weight_decay=config["WEIGHT_DECAY"]
    )
    critic_optimizer = torch.optim.Adam(
        critic.parameters(), lr=config["LR_CRITIC"], weight_decay=config["WEIGHT_DECAY"]
    )
    critic_scheduler = ExponentialLR(critic_optimizer, gamma=0.985)

    # --- 4. Baseline & Pre-checks ---
    initial_test_results_td, initial_test_results_extra = test_model(
        actor, test_problem, initial_test_solutions, config
    )
    current_test_loss = torch.mean(initial_test_results_td["min_cost"])
    logger.info(f"Baseline Test Loss: {current_test_loss:.4f}")

    # --- 5. Training Loop ---
    early_stopping_counter = 0
    best_loss_value = float("inf")
    logger.info("Starting Training Phase")

    if training_problem.device == "cuda":
        torch.cuda.empty_cache()

    progress_bar = tqdm(range(config["N_EPOCHS"]), unit="epoch", colour="blue")

    a_min_cost = current_test_loss.item()

    save_period = 5

    chain_mode = config["CL"] and config.get("CL_MODE", "warmup") == "chain"
    chain_pool = None

    for epoch in progress_bar:
        _epoch_start = time.time()
        # A. Prepare Data
        if chain_mode:
            training_problem, chain_pool, chain_solutions = refresh_chain_pool(
                training_problem, chain_pool, device, config, epoch
            )
        else:
            training_problem = initialize_training_problem(training_problem, device, config, epoch)
            chain_solutions = None

        # B. Run Training Step
        sa_td, sa_extra, train_stats, avg_actor_grad, avg_critic_grad, pre_step = train_ppo(
            actor=actor,
            critic=critic,
            actor_optimizer=actor_optimizer,
            critic_optimizer=critic_optimizer,
            critic_scheduler=critic_scheduler,
            problem=training_problem,
            config=config,
            step=epoch + 1,
            initial_solutions_override=chain_solutions,
        )
        if chain_mode:
            # Next epoch resumes each chain from this epoch's best solution
            chain_pool["solutions"] = sa_td["best_x"].detach()
        # C. Extract Stats
        actor_loss, critic_loss, avg_entropy, beta_kl, explained_var, average_kl = train_stats
        config["BETA_KL"] = beta_kl

        # D. Periodic Evaluation
        test_results_td, test_results_extra = None, None
        if epoch % save_period == 0 and epoch != 0:
            test_results_td, test_results_extra = test_model(
                actor, test_problem, initial_test_solutions, config
            )
            current_test_loss = torch.mean(test_results_td["min_cost"])
            if current_test_loss.item() < a_min_cost:
                a_min_cost = current_test_loss.item()

        # E. Early Stopping Check
        if epoch % save_period == 0:
            if current_test_loss.item() >= best_loss_value:
                early_stopping_counter += 1
            else:
                early_stopping_counter = 0
                best_loss_value = min(current_test_loss.item(), best_loss_value)

        # F. Logging
        if config["LOG"]:
            log_training_and_test_metrics(
                actor_loss=actor_loss,
                critic_loss=critic_loss,
                steps=config["OUTER_STEPS"],
                avg_actor_grad=avg_actor_grad,
                avg_critic_grad=avg_critic_grad,
                lr_actor=actor_optimizer.param_groups[0]["lr"],
                beta_kl=beta_kl,
                explained_var=explained_var,
                rewards_mean=sa_extra["average_sum_rewards"].item(),
                average_kl=average_kl,
                entropy=avg_entropy,
                pre_step=pre_step,
                early_stopping_counter=early_stopping_counter,
                a_min_cost=a_min_cost,
                test_results_td=test_results_td
                if (epoch >= save_period)
                else initial_test_results_td,
                test_results_extra=test_results_extra
                if (epoch >= save_period)
                else initial_test_results_extra,
                epoch=epoch,
                config=config,
                log_test=(epoch % save_period == 0 and epoch != 0),
            )

            # Save Checkpoint
            if epoch % save_period == 0:
                WandbLogger.log_model(
                    save_func=save_model,
                    model=actor,
                    val_loss=current_test_loss.item(),
                    epoch=epoch,
                    model_name=f"{config['PROJECT']}_{config['GROUP']}_actor",
                )
            if epoch % 50 == 0:
                WandbLogger.log_checkpoint(
                    save_func=save_model,
                    model=actor,
                    val_loss=current_test_loss.item(),
                    epoch=epoch,
                    model_name=f"{config['PROJECT']}_{config['GROUP']}_checkpoint",
                )

        # G. Loop termination
        # Early stopping is suppressed under curriculum until `early_stop_after`
        # epochs. Defaults to MAX_PROB_STEP, but can be set independently so the
        # schedule shape (MAX_PROB_STEP) and the early-stop gate are decoupled.
        early_stop_after = config["EARLY_STOP_AFTER"] or config["MAX_PROB_STEP"]
        if early_stopping_counter > 10 and (not config["CL"] or epoch >= early_stop_after):
            logger.warning(f"Early stopping triggered at epoch {epoch}")
            break

        progress_bar.set_description(
            f"Test Loss: {current_test_loss:.4f} | EarlyStop: {early_stopping_counter}"
        )

        if config.get("BENCHMARK_LOG"):
            _log_benchmark_epoch(
                config["BENCHMARK_LOG"],
                epoch,
                time.time() - _epoch_start,
                actor_loss if actor_loss is not None else float("nan"),
                critic_loss,
                current_test_loss.item(),
            )

    logger.info("Training Completed Successfully.")


if __name__ == "__main__":
    main(dict(config))
