"""
CVRP (Capacitated Vehicle Routing Problem) Solver using PPO and Simulated Annealing

This module implements a reinforcement learning approach to solve CVRP problems
using Proximal Policy Optimization (PPO) combined with Simulated Annealing for
solution refinement.

Main components:
- Actor-Critic neural networks for policy learning
- PPO for policy optimization
- Simulated Annealing for solution exploration
- WandB integration for experiment tracking
"""

# --------------------------------
# Import required libraries
# --------------------------------
import random
import warnings
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import tqdm

from algo import P_generate_instances, stack_res
from init import init_problem, initialize_models, initialize_test_problem, test_model
from model import SAModel
from ppo import ReplayBuffer, ppo
from problem import CVRP
from sa import sa_train
from setup import _HP, WandbLogger, get_script_arguments
from utils import setup_device, setup_logging, setup_reproducibility

# Logging setup
logger = setup_logging()


warnings.filterwarnings("ignore", message="Attempting to run cuBLAS")
# --------------------------------
# Load and prepare configuration
# --------------------------------
config = _HP("src/HyperParameters/HP.yaml")
config.update(get_script_arguments(config.keys()))
# --------------------------------
# Initialize experiment tracking
# --------------------------------
if config["LOG"]:
    WandbLogger.init(None, 3, config)
    logger.info(f"WandB model save directory: {WandbLogger.get_model_dir()}")


def log_training_and_test_metrics(
    actor_loss: Optional[float],
    critic_loss: float,
    avg_actor_grad: float,
    avg_critic_grad: float,
    lr_actor: float,
    beta_kl: float,
    entropy: float,
    early_stopping_counter: int,
    test_results: Optional[Dict[str, torch.Tensor]],
    epoch: int,
    config: Dict[str, Any],
) -> None:
    """
    Log training and testing metrics to WandB.

    Args:
        actor_loss: Actor network loss value
        critic_loss: Critic network loss value
        avg_actor_grad: Average gradient magnitude for actor
        avg_critic_grad: Average gradient magnitude for critic
        test_results: Dictionary containing test metrics
        epoch: Current training epoch
        config: Configuration dictionary
    """
    logs = {}

    # Log training metrics if available
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
                "Entropy": entropy,
                "early_stopping_counter": early_stopping_counter,
            }
        )

    # Log test metrics every 10 epochs
    if epoch % 10 == 0 and test_results is not None:
        test_logs = {
            "Min_cost": torch.mean(test_results["min_cost"]),
            "Gain": torch.mean(test_results["init_cost"] - test_results["min_cost"]),
            "Acceptance_rate": torch.mean(test_results["n_acc"])
            / config["TEST_OUTER_STEPS"],
            "Step_best_cost": torch.mean(test_results["best_step"])
            / config["TEST_OUTER_STEPS"],
            "Valid_percentage": torch.mean(test_results["is_valid"]),
            "Final_capacity_left": torch.mean(test_results["capacity_left"]),
        }
        logs.update(test_logs)

    WandbLogger.log(logs)


def save_model(path: str, actor_model: Optional[torch.nn.Module] = None) -> None:
    """
    Save actor model state dictionary to specified path.

    Args:
        path: Destination file path for model checkpoint
        actor_model: PyTorch model to be saved

    Raises:
        ValueError: If no model is provided for saving
    """
    if actor_model is not None:
        torch.save(actor_model.state_dict(), path)
        if config["VERBOSE"]:
            logger.info(f"Model saved to: {path}")
    else:
        raise ValueError("No model provided for saving")


def train_ppo(
    actor: SAModel,
    critic: torch.nn.Module,
    actor_optimizer: torch.optim.Optimizer,
    critic_optimizer: torch.optim.Optimizer,
    critic_scheduler: torch.optim.lr_scheduler.ExponentialLR,  # Update scheduler type
    problem: CVRP,
    initial_solutions: torch.Tensor,
    config: Dict[str, Any],
    step: int = 0,
) -> Tuple[
    Dict[str, torch.Tensor],
    Tuple[float, float, float, float],
    float,
    float,
]:
    """
    Execute one training cycle of PPO algorithm.

    Process:
    1. Collect experiences using Simulated Annealing
    2. Optimize policy using PPO
    3. Compute gradient statistics for monitoring

    Args:
        actor: Actor neural network
        critic: Critic neural network
        actor_optimizer: Optimizer for actor network
        critic_optimizer: Optimizer for critic network
        critic_scheduler: Learning rate scheduler for critic
        problem: CVRP problem instance
        initial_solutions: Initial solution tensor
        config: Configuration dictionary
        step: Current training step

    Returns:
        Tuple containing:
        - SA training results dictionary
        - Actor loss value
        - Critic loss value
        - Average actor gradient magnitude
        - Average critic gradient magnitude
    """
    # Clear GPU cache if using CUDA
    if problem.device == "cuda":
        torch.cuda.init()
        torch.cuda.empty_cache()

    # Initialize experience replay buffer
    buffer_size = config["OUTER_STEPS"] * config["INNER_STEPS"]
    replay_buffer = ReplayBuffer(buffer_size)

    # Collect experiences through Simulated Annealing
    sa_results = sa_train(
        actor=actor,
        problem=problem,
        initial_solution=initial_solutions,
        config=config,
        replay_buffer=replay_buffer,
        baseline=False,
        greedy=False,
        train=True,
    )

    # Optimize policy using PPO
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

    # Step the learning rate scheduler for the critic
    critic_scheduler.step()

    # Compute gradient statistics for monitoring
    actor_gradients = [
        param.grad.abs().mean().item()
        for param in actor.parameters()
        if param.grad is not None
    ]
    critic_gradients = [
        param.grad.abs().mean().item()
        for param in critic.parameters()
        if param.grad is not None
    ]

    avg_actor_grad = float(np.mean(actor_gradients) if actor_gradients else 0.0)
    avg_critic_grad = float(np.mean(critic_gradients) if critic_gradients else 0.0)

    # Clean up GPU memory
    if problem.device == "cuda":
        torch.cuda.empty_cache()

    return sa_results, train_stats, avg_actor_grad, avg_critic_grad


def initialize_training_problem(
    problem: CVRP, device: str, config: Dict[str, Any]
) -> CVRP:
    if config["DATA"] == "uchoa":
        # Load test problem parameters
        test_dim = config["PROBLEM_DIM"]
        n_test_problems = config["N_PROBLEMS"]
        base_seed = random.randint(0, 1000000)
        coords_list, demands_list, capacity_list, _ = P_generate_instances(
            n_test_problems, base_seed, test_dim
        )
        # Stack the results into tensors
        coords, demands, capacity = stack_res(coords_list, demands_list, capacity_list)
        # Generate and set problem parameters
        problem.generate_params("train", True, coords, demands.to(torch.int64))
        problem.capacity = capacity.to(device)
    elif config["DATA"] == "random":
        # Generate new training problem instances
        problem.generate_params()
    return problem


def main(config: dict) -> None:
    """
    Main training loop for CVRP optimization using PPO and Simulated Annealing.

    Args:
        config: Configuration dictionary containing all hyperparameters
    """
    # Setup device and reproducibility
    device = setup_device(config["DEVICE"])
    if device == "cuda":
        logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        logger.info(f"Using device: {device}")
    config["DEVICE"] = device
    setup_reproducibility(config["SEED"])
    logger.info(f"Random seeds set to: {config['SEED']}")

    training_problem, input_dim = init_problem(
        config, dim=config["PROBLEM_DIM"], n_problem=config["N_PROBLEMS"]
    )
    config["ENTRY"] = input_dim
    logger.info(f"Input dimension set to {input_dim}")

    if config["REWARD_LAST"]:
        config["REWARD_LAST_SCALE"] = 0.0

    # Initialize test problem environment
    test_problem, initial_test_solutions = initialize_test_problem(
        config,
        config["TEST_DIMENSION"],
        config["TEST_NB_PROBLEMS"],
        config["TEST_INIT"],
        "nazari" if config["NAZARI"] else "uchoa",
        device,
    )
    # Log test problem statistics
    initial_cost = torch.mean(test_problem.cost(initial_test_solutions))
    logger.info(
        f"Test problem initialized - Dimension: {config['TEST_DIMENSION']}, "
        f"Problems: {config['TEST_NB_PROBLEMS']}, Initial cost: {initial_cost.item():.2f}"
    )

    # Initialize neural network models
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
    logger.info(f"Actor model initialized: {actor.__class__.__name__}")
    logger.info("Critic model initialized")

    # Initialize optimizers
    actor_optimizer = torch.optim.Adam(
        actor.parameters(), lr=config["LR_ACTOR"], weight_decay=config["WEIGHT_DECAY"]
    )
    critic_optimizer = torch.optim.Adam(
        critic.parameters(), lr=config["LR_CRITIC"], weight_decay=config["WEIGHT_DECAY"]
    )

    # Initialize learning rate scheduler for the critic
    critic_scheduler = ExponentialLR(critic_optimizer, gamma=0.98)  # Smooth decay

    logger.info("Training initialization completed")

    # Perform initial test to establish baseline
    initial_test_results = test_model(
        actor, test_problem, initial_test_solutions, config
    )
    current_test_loss = torch.mean(initial_test_results["min_cost"])
    logger.info(
        f"Initial test completed with {config['TEST_INIT']}, "
        f"with loss: {current_test_loss:.4f}"
    )

    # Early stopping variables
    early_stopping_counter = 0
    best_loss_value = float("inf")

    logger.info("Starting training")

    # Clear GPU memory before training
    if training_problem.device == "cuda":
        torch.cuda.empty_cache()

    # Main training loop
    with tqdm(range(config["N_EPOCHS"]), unit="epoch", colour="blue") as progress_bar:
        for epoch in progress_bar:
            # Generate new training problem instances
            training_problem = initialize_training_problem(
                training_problem, device, config
            )

            # Generate initial solutions for training
            initial_training_solutions = training_problem.generate_init_state(
                config["INIT"], config["MULTI_INIT"]
            )

            # Execute training step
            training_results = train_ppo(
                actor=actor,
                critic=critic,
                actor_optimizer=actor_optimizer,
                critic_optimizer=critic_optimizer,
                critic_scheduler=critic_scheduler,  # Pass scheduler
                problem=training_problem,
                initial_solutions=initial_training_solutions,
                config=config,
                step=epoch + 1,
            )

            # Unpack training results
            (
                sa_results,
                train_stats,
                avg_actor_grad,
                avg_critic_grad,
            ) = training_results

            actor_loss, critic_loss, avg_entropy, beta_kl = train_stats

            config["BETA_KL"] = beta_kl  # Update beta_kl from PPO

            # Periodic testing and evaluation
            test_results = None
            if epoch % 10 == 0 and epoch != 0:
                test_results = test_model(
                    actor, test_problem, initial_test_solutions, config
                )
                current_test_loss = torch.mean(test_results["min_cost"])

                if config["REWARD_LAST"]:
                    config["REWARD_LAST_SCALE"] = min(
                        config["REWARD_LAST_SCALE"] + config["REWARD_LAST_ADD"], 100
                    )

            # Early stopping logic
            if epoch % 10 == 0:
                if current_test_loss.item() >= best_loss_value:
                    early_stopping_counter += 1
                    if config["VERBOSE"]:
                        logger.info(f"Early stopping counter: {early_stopping_counter}")
                else:
                    early_stopping_counter = 0
                    best_loss_value = min(current_test_loss.item(), best_loss_value)

            # Log metrics to WandB
            if config["LOG"]:
                lr_actor = actor_optimizer.param_groups[0]["lr"]
                log_training_and_test_metrics(
                    actor_loss=actor_loss,
                    critic_loss=critic_loss,
                    avg_actor_grad=avg_actor_grad,
                    avg_critic_grad=avg_critic_grad,
                    lr_actor=lr_actor,
                    beta_kl=beta_kl,
                    entropy=avg_entropy,
                    early_stopping_counter=early_stopping_counter,
                    test_results=(
                        test_results if (epoch >= 10) else initial_test_results
                    ),
                    epoch=epoch,
                    config=config,
                )

            # Model checkpointing
            if config["LOG"] and epoch % 10 == 0:
                model_name = f"{config['PROJECT']}_{config['GROUP']}_actor"
                WandbLogger.log_model(
                    save_func=save_model,
                    model=actor,
                    val_loss=current_test_loss.item(),
                    epoch=epoch,
                    model_name=model_name,
                )

            # Trigger early stopping if no improvement for too long
            if early_stopping_counter > 10:
                logger.warning(
                    f"Early stopping triggered at epoch {epoch} "
                    f"with loss {best_loss_value:.4f}"
                )
                break

            # Update progress bar
            progress_bar.set_description(
                (
                    f"Test loss: {current_test_loss:.4f}, "
                    f"EarlyStop Counter: {early_stopping_counter}"
                )
            )

    logger.info("Training completed successfully")


if __name__ == "__main__":
    main(dict(config))
