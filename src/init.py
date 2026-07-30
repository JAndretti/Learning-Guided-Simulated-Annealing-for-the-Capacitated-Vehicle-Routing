import pickle
from typing import Any

import torch

from model import CVRPActor, CVRPActorShared, CVRPCritic, CVRPCriticDeepSets, SAModel
from problem import CVRP
from sa import sa_test, sa_train


def init_problem(config: dict, dim: int, n_problem: int) -> tuple[CVRP, int]:
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
    config: dict,
    baseline: bool = False,
    greedy: bool = False,
) -> dict[str, torch.Tensor]:
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

    problem.init_parameters(initial_solutions)

    # Perform Simulated Annealing for testing
    results_td, results_extra = sa_train(
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

    return results_td, results_extra


def inf_test_model(
    actor: SAModel,
    problem: CVRP,
    initial_solutions: torch.Tensor,
    config: dict,
    baseline: bool = False,
    greedy: bool = False,
    dtype: torch.dtype = torch.float32,
) -> dict[str, torch.Tensor]:
    """
    Test the trained model performance using Simulated Annealing with fast inference (less metrics etc..).

    Args:
        actor: Trained actor network
        problem: CVRP problem instance for testing
        initial_solutions: Initial solution tensor
        config: Configuration dictionary
        baseline: Whether to use baseline mode in testing
        greedy: Whether to use greedy mode in testing
        dtype: Floating-point precision for the full SA loop

    Returns:
        Dictionary containing test results and metrics
    """

    # Clear GPU cache if using CUDA
    if problem.device == "cuda":
        torch.cuda.empty_cache()

    problem.init_parameters(initial_solutions)

    # Perform Simulated Annealing for testing
    results_td, results_extra = sa_test(
        actor=actor,
        problem=problem,
        initial_solution=initial_solutions,
        config=config,
        baseline=baseline,
        greedy=greedy,
        dtype=dtype,
    )

    # Clear GPU cache if using CUDA
    if problem.device == "cuda":
        torch.cuda.empty_cache()

    return results_td, results_extra


def load_neuopt_nazari(test_dim: int, n_problems: int) -> tuple[torch.Tensor, ...]:
    """Load the NeuOpt-distributed nazari test set and convert it to our tensor layout.

    The pickle holds a list of `(depot_xy, node_xy, demands, capacity)` tuples with the
    same conventions as our own `.pt` files (coords in [0, 1], integer demands).

    Args:
        test_dim: Number of customers (depot excluded)
        n_problems: Number of instances to keep

    Returns:
        Tuple of (node_coords [B, dim+1, 2], demands [B, dim+1], capacity [B, 1])
    """
    path = f"generated_nazari_problem/NeuOpt_data/cvrp_{test_dim}.pkl"
    try:
        with open(path, "rb") as f:
            raw = pickle.load(f)
    except FileNotFoundError:
        print(f"NeuOpt test data file not found: {path}")
        raise

    raw = raw[:n_problems]
    depot = torch.tensor([inst[0] for inst in raw], dtype=torch.float32)  # [B, 2]
    nodes = torch.tensor([inst[1] for inst in raw], dtype=torch.float32)  # [B, dim, 2]
    dem = torch.tensor([inst[2] for inst in raw], dtype=torch.int64)  # [B, dim]
    cap = torch.tensor([inst[3] for inst in raw], dtype=torch.int64)  # [B]

    coordinates = torch.cat([depot.unsqueeze(1), nodes], dim=1)
    demands = torch.cat([torch.zeros_like(dem[:, :1]), dem], dim=1)
    capacities = cap.unsqueeze(1)
    return coordinates, demands, capacities


def initialize_test_problem(
    config: dict[str, Any],
    test_dim: int,
    n_test_problems: int,
    init_method: str,
    data: str = "nazari",
    device: str = "cpu",
    source: str = "default",
    offset: int = 0,
) -> tuple[CVRP, torch.Tensor]:
    """
    Initialize test problem instance with pre-generated data.

    Args:
        config: Configuration dictionary
        test_dim: Dimension of the test problem
        n_test_problems: Number of test problem instances
        init_method: Method for generating initial solutions
        data: Dataset type ("nazari" or "uchoa")
        device: Compute device string
        source: For `data="nazari"`, which test set to read — "default" (our own
            `gen_nazari_{dim}.pt`) or "neuopt" (`NeuOpt_data/cvrp_{dim}.pkl`)
        offset: Index of the first instance to keep, so disjoint slices of one
            file can be used as separate sets (e.g. a reported set at offset 0
            and a tuning set at offset 1000). Only honoured for
            `data="nazari", source="default"`; a non-zero offset with any other
            combination raises, rather than being silently ignored.

    Returns:
        Tuple of (test_problem_instance, initial_test_solutions)
    """
    if offset != 0 and not (data == "nazari" and source == "default"):
        raise ValueError(
            f"offset={offset} is only supported for data='nazari', source='default' "
            f"(got data={data!r}, source={source!r})"
        )

    if data == "nazari":
        if source == "neuopt":
            coordinates, demands, capacities = load_neuopt_nazari(
                test_dim, n_test_problems
            )
            coordinates = coordinates.to(device)
            demands = demands.to(device)
            capacities = capacities.to(device)
        elif source == "default":
            path = f"generated_nazari_problem/gen_nazari_{test_dim}.pt"

            try:
                test_data = torch.load(path, map_location="cpu")
            except FileNotFoundError:
                print(f"Nazari test data file not found: {path}")
                raise
            lo, hi = offset, offset + n_test_problems
            available = test_data["node_coords"].shape[0]
            if hi > available:
                raise ValueError(
                    f"Requested instances [{lo}:{hi}] but {path} holds only {available}"
                )
            coordinates = test_data["node_coords"][lo:hi].to(device)
            demands = test_data["demands"][lo:hi].to(device)
            capacities = test_data["capacity"][lo:hi].to(device)
        else:
            raise ValueError(f"Unknown nazari data source: {source}")

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
    test_problem.generate_params(coordinates, demands, capacities)

    initial_test_solutions = test_problem.generate_init_state(init_method, False)

    return test_problem, initial_test_solutions


def initialize_models(
    model_type: str,
    critic_type: str,
    embedding_dim: int,
    entry: int,
    num_h_layers: int,
    update_method: str,
    heuristic: str | list[str],
    seed: int = 0,
    device: str = "cpu",
    cond_rank: bool = False,
    cond_detour: bool = False,
    global_context: bool = False,
    bilinear: bool = False,
    logit_clip: float = 0.0,
    learnable_temp: bool = False,
) -> tuple[SAModel, CVRPCritic | CVRPCriticDeepSets]:
    """
    Initialize actor and critic neural networks.

    Args:
        model_type: Type of actor model ("seq" or "shared")
        critic_type: Type of critic model ("ff" or "deepsets")
        embedding_dim: Dimension of embeddings
        entry: Number of input features
        num_h_layers: Number of hidden layers
        update_method: Method for updating heuristic information
        heuristic: Heuristic(s) to be used
        seed: Random seed for model initialization
        device: Compute device string
        cond_rank: Condition city-2 selection on bidirectional edge ranks
        cond_detour: Condition city-2 selection on insertion cost
        global_context: Feed a mean|max|std pooled instance summary to both stages
        bilinear: Bilinear compatibility stage-2 scorer ("shared" actor only)
        logit_clip: 0 disables; >0 applies C*tanh(logits) clipping
        learnable_temp: Learnable softmax temperature, one per stage

    Returns:
        Tuple of (actor_model, critic_model)
    """
    # Determine if mixed heuristic is used
    if isinstance(heuristic, str):
        heuristic = [heuristic]

    # Initialize actor model
    if model_type == "seq":
        actor = CVRPActor(
            embed_dim=embedding_dim,
            c=entry,
            num_hidden_layers=num_h_layers,
            device=device,
            method=update_method,
            cond_rank=cond_rank,
            cond_detour=cond_detour,
            global_context=global_context,
            logit_clip=logit_clip,
            learnable_temp=learnable_temp,
        )
    elif model_type == "shared":
        actor = CVRPActorShared(
            embed_dim=embedding_dim,
            c=entry,
            num_hidden_layers=num_h_layers,
            device=device,
            method=update_method,
            cond_rank=cond_rank,
            cond_detour=cond_detour,
            global_context=global_context,
            bilinear=bilinear,
            logit_clip=logit_clip,
            learnable_temp=learnable_temp,
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
    elif critic_type == "deepsets":
        critic = CVRPCriticDeepSets(
            embed_dim=embedding_dim,
            c=entry,
            num_hidden_layers=num_h_layers,
            device=device,
        )
    else:
        raise ValueError(f"Unknown critic model type specified: {critic_type}")

    return actor, critic
