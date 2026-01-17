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
import random
import warnings
from typing import Any, Dict, Optional, Tuple

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
    actor_loss: Optional[float],
    critic_loss: float,
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
    test_results: Optional[Dict[str, torch.Tensor]],
    epoch: int,
    config: Dict[str, Any],
) -> None:
    """Logs training and testing metrics to WandB."""
    logs = {}

    # 1. Training Metrics
    if actor_loss is not None:
        logs.update(
            {
                "Actor_loss": actor_loss,
                "Critic_loss": critic_loss,
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
    if epoch % 10 == 0 and test_results is not None:
        logs.update(
            {
                "Min_cost": torch.mean(test_results["min_cost"]),
                "A_min_cost": a_min_cost,
                "Gain": torch.mean(
                    test_results["init_cost"] - test_results["min_cost"]
                ),
                "Test_Rewards_mean": test_results["average_sum_rewards"].item(),
                "Acceptance_rate": torch.mean(test_results["n_acc"])
                / config["TEST_OUTER_STEPS"],
                "Step_best_cost": torch.mean(test_results["best_step"])
                / config["TEST_OUTER_STEPS"],
                "Valid_percentage": torch.mean(test_results["is_valid"]),
                "Final_capacity_left": torch.mean(test_results["capacity_left"]),
            }
        )

    WandbLogger.log(logs)


def save_model(path: str, actor_model: Optional[torch.nn.Module] = None) -> None:
    """Saves the actor model state dictionary."""
    if actor_model is None:
        raise ValueError("No model provided for saving")

    torch.save(actor_model.state_dict(), path)
    if config["VERBOSE"]:
        logger.info(f"Model saved to: {path}")


def calculate_curriculum_steps(step: int, config: Dict[str, Any]) -> int:
    """
    Calculates the number of deterministic improvement steps (T_init)
    based on a Sigmoid schedule.
    """
    xi_cl = config["MAX_OUTER_STEPS_CL"]  # Max curriculum steps
    E = config["MAX_PROB_STEP"]  # Total curriculum epochs

    # Scale kappa based on total steps to maintain curve shape relative to paper
    # Paper used kappa=0.2 for 200 epochs
    kappa_base = 0.2
    epochs_paper = 100
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


def initialize_training_problem(
    problem: CVRP, device: str, config: Dict[str, Any]
) -> CVRP:
    """Regenerates the problem instance for the next training epoch."""
    if config["DATA"] == "uchoa":
        # Load structured instances
        coords_list, demands_list, capacity_list, _ = P_generate_instances(
            config["N_PROBLEMS"], random.randint(0, 1000000), config["PROBLEM_DIM"]
        )
        coords, demands, capacity = stack_res(coords_list, demands_list, capacity_list)
        problem.generate_params(coords, demands.to(torch.int64), capacity)

    elif config["DATA"] == "random":
        # Generate fully random instances
        coords = torch.rand(
            config["N_PROBLEMS"], config["PROBLEM_DIM"] + 1, 2, device=device
        )
        demands = torch.randint(
            1, 10, (config["N_PROBLEMS"], config["PROBLEM_DIM"] + 1), device=device
        )
        demands[:, 0] = 0  # Depot has no demand
        capacity = torch.full(
            (config["N_PROBLEMS"], 1), config["MAX_LOAD"], device=device
        )
        problem.generate_params(coords, demands, capacity)

    return problem


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
    config: Dict[str, Any],
    step: int = 0,
) -> Tuple[Dict, Tuple, float, float, int]:
    """
    Executes a single training epoch (SA collection + PPO update).
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

    # Generate initial solutions with the dynamic list
    initial_solutions = problem.generate_init_state(
        init_heuristic=config["INIT"],
        multi_init=config["MULTI_INIT"],
        init_list=current_init_list,  # Pass the dynamic list here
    )
    buffer_size = config["OUTER_STEPS"]
    replay_buffer = ReplayBuffer(buffer_size)
    pre_step = 0

    # 2. Curriculum Learning (Improvement Phase)

    if config["CL"]:
        t_init = calculate_curriculum_steps(step, config)

        if t_init > 0:
            original_steps = config.get("TEST_OUTER_STEPS", 0)
            config["TEST_OUTER_STEPS"] = t_init
            pre_step = t_init

            # Run deterministic improvement
            pre_res = sa_train(
                actor=actor,
                problem=problem,
                initial_solution=initial_solutions,
                config=config,
                replay_buffer=None,
                baseline=False,
                greedy=False,
                train=False,
            )

            # Use improved solutions as start point for training
            config["TEST_OUTER_STEPS"] = original_steps
            initial_solutions = pre_res["best_x"].detach()
            problem.init_parameters(initial_solutions)
            del pre_res

    # 3. Experience Collection (Simulated Annealing)
    sa_results = sa_train(
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
        grads = [
            p.grad.abs().mean().item() for p in model.parameters() if p.grad is not None
        ]
        return float(np.mean(grads)) if grads else 0.0

    avg_actor_grad = get_avg_grad(actor)
    avg_critic_grad = get_avg_grad(critic)

    if problem.device == "cuda":
        torch.cuda.empty_cache()

    return sa_results, train_stats, avg_actor_grad, avg_critic_grad, pre_step


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

    setup_reproducibility(config["SEED"])
    logger.info(f"Random Seed: {config['SEED']}")

    training_problem, input_dim = init_problem(
        config, dim=config["PROBLEM_DIM"], n_problem=config["N_PROBLEMS"]
    )
    config["ENTRY"] = input_dim
    logger.info(f"Problem Input Dimension: {input_dim}")

    if config["REWARD_LAST"]:
        config["REWARD_LAST_SCALE"] = 0.0

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
    initial_test_results = test_model(
        actor, test_problem, initial_test_solutions, config
    )
    current_test_loss = torch.mean(initial_test_results["min_cost"])
    logger.info(f"Baseline Test Loss: {current_test_loss:.4f}")

    # --- 5. Training Loop ---
    early_stopping_counter = 0
    best_loss_value = float("inf")
    logger.info("Starting Training Phase")

    if training_problem.device == "cuda":
        torch.cuda.empty_cache()

    progress_bar = tqdm(range(config["N_EPOCHS"]), unit="epoch", colour="blue")

    a_min_cost = current_test_loss.item()

    for epoch in progress_bar:
        # A. Prepare Data
        training_problem = initialize_training_problem(training_problem, device, config)

        # B. Run Training Step
        sa_results, train_stats, avg_actor_grad, avg_critic_grad, pre_step = train_ppo(
            actor=actor,
            critic=critic,
            actor_optimizer=actor_optimizer,
            critic_optimizer=critic_optimizer,
            critic_scheduler=critic_scheduler,
            problem=training_problem,
            config=config,
            step=epoch + 1,
        )

        # C. Extract Stats
        actor_loss, critic_loss, avg_entropy, beta_kl, explained_var, average_kl = (
            train_stats
        )
        config["BETA_KL"] = beta_kl

        # D. Periodic Evaluation
        test_results = None
        if epoch % 10 == 0 and epoch != 0:
            test_results = test_model(
                actor, test_problem, initial_test_solutions, config
            )
            current_test_loss = torch.mean(test_results["min_cost"])
            if current_test_loss.item() < a_min_cost:
                a_min_cost = current_test_loss.item()

            if config["REWARD_LAST"]:
                config["REWARD_LAST_SCALE"] = min(
                    config["REWARD_LAST_SCALE"] + config["REWARD_LAST_ADD"], 100
                )

        # E. Early Stopping Check
        if epoch % 10 == 0:
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
                avg_actor_grad=avg_actor_grad,
                avg_critic_grad=avg_critic_grad,
                lr_actor=actor_optimizer.param_groups[0]["lr"],
                beta_kl=beta_kl,
                explained_var=explained_var,
                rewards_mean=sa_results["average_sum_rewards"].item(),
                average_kl=average_kl,
                entropy=avg_entropy,
                pre_step=pre_step,
                early_stopping_counter=early_stopping_counter,
                a_min_cost=a_min_cost,
                test_results=test_results if (epoch >= 10) else initial_test_results,
                epoch=epoch,
                config=config,
            )

            # Save Checkpoint
            if epoch % 10 == 0:
                WandbLogger.log_model(
                    save_func=save_model,
                    model=actor,
                    val_loss=current_test_loss.item(),
                    epoch=epoch,
                    model_name=f"{config['PROJECT']}_{config['GROUP']}_actor",
                )

        # G. Loop termination
        if early_stopping_counter > 10 and (
            not config["CL"] or epoch >= config["MAX_PROB_STEP"]
        ):
            logger.warning(f"Early stopping triggered at epoch {epoch}")
            break

        progress_bar.set_description(
            f"Test Loss: {current_test_loss:.4f} | EarlyStop: {early_stopping_counter}"
        )

    logger.info("Training Completed Successfully.")


if __name__ == "__main__":
    main(dict(config))
