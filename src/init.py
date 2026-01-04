from typing import Any, Dict, List, Tuple, Union

import torch

from model import CVRPActor, CVRPActorPairs, CVRPCritic, SAModel
from problem import CVRP
from sa import sa_test, sa_train


def init_problem(config: dict, dim: int, n_problem: int) -> Tuple[CVRP, int]:
    """Initialize the CVRP problem environment based on the provided configuration.

    Args:
        config (dict): Configuration dictionary containing problem parameters.
        dim (int): Dimension of the problem.
        n_problem (int): Number of problem instances to generate.

    Returns:
        CVRP: An instance of the CVRP problem environment.
        input_dim (int): The input dimension for the model.
    """
    # Initialize  problem environment
    problem = CVRP(
        dim=dim,
        n_problems=n_problem,
        device=config["DEVICE"],
        params=config,
    )
    problem.manual_seed(config["SEED"])
    if "HEURISTIC" in config:
        problem.set_heuristic(config["HEURISTIC"])
    input_dim = 0
    if "features" in config:
        problem.set_feature_flags(config["features"])
        input_dim = problem.get_input_dim()
    return problem, input_dim


def test_model(
    actor: SAModel,
    problem: CVRP,
    initial_solutions: torch.Tensor,
    config: Dict,
    baseline: bool = False,
    greedy: bool = False,
) -> Dict[str, torch.Tensor]:
    """
    Test the trained model performance using Simulated Annealing.

    Args:
        actor: Trained actor network
        problem: CVRP problem instance for testing
        initial_solutions: Initial solution tensor
        config: Configuration dictionary
        baseline: Whether to use baseline mode in testing
        greedy: Whether to use greedy selection during testing

    Returns:
        Dictionary containing test results and metrics
    """
    # Clear GPU cache if using CUDA
    if problem.device == "cuda":
        torch.cuda.empty_cache()

    # Perform Simulated Annealing for testing
    test_results = sa_train(
        actor=actor,
        problem=problem,
        initial_solution=initial_solutions,
        config=config,
        replay_buffer=None,
        baseline=baseline,
        greedy=greedy,
        train=False,
    )

    # Clean up GPU memory
    if problem.device == "cuda":
        torch.cuda.empty_cache()

    return test_results


def inf_test_model(
    actor: SAModel,
    problem: CVRP,
    initial_solutions: torch.Tensor,
    config: Dict,
    baseline: bool = False,
    greedy: bool = False,
) -> Dict[str, torch.Tensor]:
    """
    Test the trained model performance using Simulated Annealing with fast inference (less metrics etc..).

    Args:
        actor: Trained actor network
        problem: CVRP problem instance for testing
        initial_solutions: Initial solution tensor
        config: Configuration dictionary
        baseline: Whether to use baseline mode in testing
        greedy: Whether to use greedy mode in testing

    Returns:
        Dictionary containing test results and metrics
    """

    # Perform Simulated Annealing for testing
    test_results = sa_test(
        actor=actor,
        problem=problem,
        initial_solution=initial_solutions,
        config=config,
        baseline=baseline,
        greedy=greedy,
    )

    return test_results


def initialize_test_problem(
    config: Dict[str, Any],
    test_dim: int,
    n_test_problems: int,
    init_method: str,
    data: str = "nazari",
    device: str = "cpu",
) -> Tuple[CVRP, torch.Tensor]:
    """
    Initialize test problem instance with pre-generated data.

    Args:
        config: Configuration dictionary
        test_dim: Dimension of the test problem
        n_test_problems: Number of test problem instances
        init_method: Method for generating initial solutions
        data: Dataset type ("nazari" or "uchoa")
        device: Compute device string

    Returns:
        Tuple of (test_problem_instance, initial_test_solutions)
    """

    if data == "nazari":
        path = f"generated_nazari_problem/gen_nazari_{test_dim}.pt"

        try:
            test_data = torch.load(path, map_location="cpu")
        except FileNotFoundError:
            print(f"Nazari test data file not found: {path}")
            raise
        coordinates = test_data["node_coords"][:n_test_problems].to(device)
        demands = test_data["demands"][:n_test_problems].to(device)
        capacities = test_data["capacity"][:n_test_problems].to(device)

    elif data == "uchoa":
        problem_path = f"generated_uchoa_problem/gen_uchoa_{test_dim}.pt"
        try:
            test_data = torch.load(problem_path, map_location="cpu")
        except FileNotFoundError:
            print(f"Test data file not found: {problem_path}")
            raise
        # Randomly select test problem indices
        indices = torch.randperm(n_test_problems, generator=torch.Generator())
        coordinates = test_data["node_coords"][indices]
        demands = test_data["demands"][indices]
        capacities = test_data["capacity"][indices]
    else:
        raise ValueError(f"Unknown data type: {data}")

    # Initialize test problem instance
    test_problem, _ = init_problem(config, dim=test_dim, n_problem=n_test_problems)

    # Generate and set problem parameters
    test_problem.generate_params("test", True, coordinates, demands, capacities)

    initial_test_solutions = test_problem.generate_init_state(init_method, False)

    return test_problem, initial_test_solutions


def initialize_models(
    model_type: str,
    critic_type: str,
    embedding_dim: int,
    entry: int,
    num_h_layers: int,
    update_method: str,
    heuristic: Union[str, List[str]],
    seed: int = 0,
    device: str = "cpu",
) -> Tuple[SAModel, CVRPCritic]:
    """
    Initialize actor and critic neural networks.

    Args:
        model_type: Type of actor model ("pairs" or "seq")
        critic_type: Type of critic model ("ff")
        embedding_dim: Dimension of embeddings
        entry: Number of input features
        num_h_layers: Number of hidden layers
        update_method: Method for updating heuristic information
        heuristic: Heuristic(s) to be used
        seed: Random seed for model initialization
        train: Whether to initialize the critic model
        device: Compute device string

    Returns:
        Tuple of (actor_model, critic_model)
    """
    # Determine if mixed heuristic is used
    if isinstance(heuristic, str):
        heuristic = [heuristic]
    use_mixed_heuristic = len(heuristic) > 1

    # Initialize actor model (with or without pairs)
    if model_type == "pairs":
        actor = CVRPActorPairs(
            embed_dim=embedding_dim,
            c=entry,
            num_hidden_layers=num_h_layers,
            device=device,
            mixed_heuristic=use_mixed_heuristic,
            method=update_method,
        )
    elif model_type == "seq":
        actor = CVRPActor(
            embed_dim=embedding_dim,
            c=entry,
            num_hidden_layers=num_h_layers,
            device=device,
            mixed_heuristic=use_mixed_heuristic,
            method=update_method,
        )
    else:
        raise ValueError(f"Unknown model type specified: {model_type}")

    actor.manual_seed(seed)
    # Initialize critic model
    if critic_type == "ff":
        critic = CVRPCritic(
            embed_dim=embedding_dim,
            c=entry,
            num_hidden_layers=num_h_layers,
            device=device,
        )
    else:
        raise ValueError(f"Unknown critic model type specified: {critic_type}")

    return actor, critic
