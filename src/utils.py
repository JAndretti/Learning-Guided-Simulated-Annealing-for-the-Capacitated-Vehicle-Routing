import os
import random
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from loguru import logger
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle


def setup_logging():
    # 1. Clear default handlers
    logger.remove()

    # --- CUSTOM COLORS SETUP ---
    # We explicitly tell loguru what colors to use for specific levels.
    # <yellow> is standard, but you can also use hex codes like <#FFA500> for orange if your terminal supports it.
    logger.level("INFO", color="<yellow>")  # Yellow/Orange for INFO
    logger.level("WARNING", color="<red>")  # Red for WARNING
    logger.level("ERROR", color="<magenta>")  # Bold Red for ERROR

    # 2. Console Handler
    # Notice the format ends with: <level>{message}</level>
    # This tells loguru to paint the message text with the level's color defined above.
    logger.add(
        sys.stderr,
        level="INFO",
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{file}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    )
    return logger


def setup_device(device: str) -> str:
    """
    Configure compute device and logging based on availability.

    Args:
        device: Desired device string ('cuda', 'mps', or 'cpu')

    Returns:
        Selected device string ('cuda', 'mps', or 'cpu')
    """

    # CUDA device configuration
    if "cuda" in device:
        if torch.cuda.is_available():
            os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
            torch.cuda.empty_cache()
            torch.backends.cudnn.benchmark = True

    # Apple Metal Performance Shaders (MPS) configuration
    else:
        if torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    return device


def setup_reproducibility(seed: int) -> None:
    """
    Set random seeds for reproducible results across different runs.

    Args:
        seed: Random seed value
    """
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.

    # Ensure deterministic behavior (optional but recommended for full reproducibility)
    # This makes operations like convolution deterministic but might slow down training slightly.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def extend(tensor: torch.Tensor, dims: int) -> torch.Tensor:
    """Extend tensor to match dimensions 'dims'."""
    return tensor[(...,) + (None,) * dims]


def extend_to(tensor1: torch.Tensor, tensor2: torch.Tensor) -> torch.Tensor:
    """Extend tensor1 to have same number of dims as tensor2."""
    return extend(tensor1, len(tensor2.shape) - len(tensor1.shape))


def repeat_to(tensor1: torch.Tensor, tensor2: torch.Tensor) -> torch.Tensor:
    """Repeat tensor1 to match the shape of tensor2."""
    tensor1 = extend_to(tensor1, tensor2)
    ones = torch.ones(tensor2.shape[:-1] + (1,), device=tensor1.device)
    return tensor1 * ones


def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Convert torch tensor to numpy array."""
    return tensor.detach().cpu().numpy()


# def init_problem(config: dict, dim: int, n_problem: int) -> Tuple[CVRP, int]:
#     """Initialize the CVRP problem environment based on the provided configuration.

#     Args:
#         config (dict): Configuration dictionary containing problem parameters.

#     Returns:
#         CVRP: An instance of the CVRP problem environment.
#         input_dim (int): The input dimension for the model.
#     """
#     # Initialize  problem environment
#     problem = CVRP(
#         dim=dim,
#         n_problems=n_problem,
#         device=config["DEVICE"],
#         params=config,
#     )
#     problem.manual_seed(config["SEED"])
#     problem.set_heuristic(config["HEURISTIC"])

#     problem.set_feature_flags(config["features"])
#     input_dim = problem.get_input_dim()
#     return problem, input_dim


def convert_tensor(a: torch.Tensor) -> torch.Tensor:
    """Convert a tensor to a tensor of indices where values change."""
    # Create a mask to identify indices where values change
    diff = (a[:, 1:] != a[:, :-1]) & (a[:, 1:] != 0)

    # Initialize a result tensor with zeros
    result = torch.zeros_like(a, dtype=torch.int)

    # Fill the first column of result with zeros
    result[:, 0] = 0

    # Use a tensor to increment indices where there is a non-zero value change
    indices = torch.cumsum(diff, dim=1)
    # Add the last value of each row to indices
    indices = torch.cat((indices, indices[:, -1].unsqueeze(1)), dim=1)
    return indices


# Code inspired by Google OR Tools plot:
# https://github.com/google/or-tools/blob/fb12c5ded7423d524fc6c95656a9bdc290a81d4d/examples/python/cvrptw_plot.py


def discrete_cmap(N, base_cmap=None):
    """
    Create an N-bin discrete colormap from the specified input map
    """
    # Note that if base_cmap is a string or None, you can simply do
    #    return plt.cm.get_cmap(base_cmap, N)
    # The following works for string, None, or a colormap instance:

    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)
    return base.from_list(cmap_name, color_list, N)


def prepare_plot(pb, init):
    depot = pb.coords[:, 0, :].squeeze().cpu()
    loc = pb.coords[:, 1:, :].squeeze().cpu()
    demand = pb.demands[:, 1:].squeeze().cpu()
    data = {"depot": depot, "loc": loc, "demand": demand}
    init = init.squeeze().cpu()
    return data, init


def plot_vehicle_routes(
    data,
    route,
    ax1,
    markersize=5,
    visualize_demands=False,
    capacity=1,
    round_demand=False,
    title="None",
):
    """
    Plot the vehicle routes on matplotlib axis ax1.
    data: dict with 'depot', 'loc', 'demand'
    route: LongTensor [route_length] with 0 for depot and 1..N
    ax1: matplotlib axis
    """

    # route is one sequence, separating different routes with 0 (depot)
    routes = [
        r[r != 0]
        for r in np.split(route.cpu().numpy(), np.where(route == 0)[0])
        if (r != 0).any()
    ]
    depot = data["depot"].cpu().numpy()
    locs = data["loc"].cpu().numpy()
    demands = data["demand"].cpu().numpy()  # * demand_scale

    x_dep, y_dep = depot
    ax1.plot(x_dep, y_dep, "sk", markersize=markersize * 4)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)

    # ax1.legend(loc="upper center")

    cmap = discrete_cmap(len(routes) + 2, "nipy_spectral")
    dem_rects = []
    used_rects = []
    cap_rects = []
    qvs = []
    total_dist = 0
    for veh_number, r in enumerate(routes):
        color = cmap(len(routes) - veh_number)  # Invert to have in rainbow order

        route_demands = demands[r - 1]
        coords = locs[r - 1, :]
        xs, ys = coords.transpose()

        total_route_demand = sum(route_demands)
        assert total_route_demand <= capacity
        if not visualize_demands:
            ax1.plot(xs, ys, "o", mfc=color, markersize=markersize, markeredgewidth=0.0)

        dist = 0
        x_prev, y_prev = x_dep, y_dep
        cum_demand = 0
        for (x, y), d in zip(coords, route_demands):
            dist += np.sqrt((x - x_prev) ** 2 + (y - y_prev) ** 2)

            cap_rects.append(Rectangle((x, y), 0.01, 0.1))
            used_rects.append(
                Rectangle((x, y), 0.01, 0.1 * total_route_demand / capacity)
            )
            dem_rects.append(
                Rectangle(
                    (x, y + 0.1 * cum_demand / capacity), 0.01, 0.1 * d / capacity
                )
            )

            x_prev, y_prev = x, y
            cum_demand += d

        dist += np.sqrt((x_dep - x_prev) ** 2 + (y_dep - y_prev) ** 2)
        total_dist += dist
        qv = ax1.quiver(
            xs[:-1],
            ys[:-1],
            xs[1:] - xs[:-1],
            ys[1:] - ys[:-1],
            scale_units="xy",
            angles="xy",
            scale=1,
            color=color,
            label="R{}, # {}, c {} / {}, d {:.2f}".format(
                veh_number,
                len(r),
                int(total_route_demand) if round_demand else total_route_demand,
                int(capacity) if round_demand else capacity,
                dist,
            ),
        )

        qvs.append(qv)
    title = (
        f"{title or ''} {len(routes)} routes, total distance {total_dist:.4f}".strip()
    )

    ax1.set_title(title)
    ax1.legend(handles=qvs)

    pc_cap = PatchCollection(
        cap_rects, facecolor="whitesmoke", alpha=1.0, edgecolor="lightgray"
    )
    pc_used = PatchCollection(
        used_rects, facecolor="lightgray", alpha=1.0, edgecolor="lightgray"
    )
    pc_dem = PatchCollection(dem_rects, facecolor="black", alpha=1.0, edgecolor="black")

    if visualize_demands:
        ax1.add_collection(pc_cap)
        ax1.add_collection(pc_used)
        ax1.add_collection(pc_dem)


def calculate_client_angles(coords: torch.Tensor) -> torch.Tensor:
    """
    coords: [B, N, 2], coords[:,0] = depot
    returns [B, N, 1] with normalized angle [0,1] (depot=0)
    """
    depot = coords[:, :1]
    clients = coords[:, 1:]
    delta = clients - depot
    ang = torch.atan2(delta[..., 1], delta[..., 0])  # [-π,π]
    norm = ang.div(2 * torch.pi).add(0.5)  # [0,1]
    all_ang = torch.cat(
        [torch.zeros(coords.size(0), 1, device=coords.device), norm], dim=1
    )
    return all_ang.unsqueeze(-1)


def calculate_distance_matrix(coords: torch.Tensor) -> torch.Tensor:
    """
    coords: [B, N, 2] → Euclidean distances [B, N, N]
    """
    return torch.cdist(coords, coords, p=2)


def calculate_knn_isolation(dists: torch.Tensor, k: int = 5) -> torch.Tensor:
    """
    Computes the average distance to the k-nearest neighbors.

    Args:
        dists: [B, N, N] Distance matrix
        k: Number of neighbors to consider

    Returns:
        [B, N, 1] Isolation score (higher = more isolated)
    """
    # We want the k smallest distances.
    # We set k=k+1 because the closest 'neighbor' is the node itself (dist=0),
    # which we want to ignore.
    # largest=False gives us the smallest values.
    values, _ = torch.topk(dists, k=k + 1, dim=-1, largest=False)

    # values shape is [B, N, k+1]
    # We slice [:, :, 1:] to drop the first column (distance to self, which is 0)
    knn_dists = values[:, :, 1:]

    # Calculate average distance to these k neighbors
    # Shape becomes [B, N, 1] to match your feature dimensions
    score = knn_dists.mean(dim=-1, keepdim=True)

    return score


def calculate_detour_features(
    solution: torch.Tensor,
    distance_matrix: torch.Tensor,
) -> torch.Tensor:
    """
    Computes the detour cost for each node in the current solution.
    Detour = (d_prev_curr + d_curr_next) - d_prev_next

    Args:
        solution: [B, L, 1] containing node indices (LongTensor)
        distance_matrix: [B, N, N] containing pairwise distances

    Returns:
        detour_features: [B, N, 1] containing the cost saving for removing each node.
    """
    # 1. Prepare shapes and remove last dimension if present
    # Shape: [B, L, 1] -> [B, L]
    sol = solution.squeeze(-1).long()
    B, L = sol.shape

    # 2. Identify Previous and Next nodes for every position
    # We roll the solution to align neighbors with the current node
    # prev_sol[b, i] is the node visited BEFORE sol[b, i]
    # next_sol[b, i] is the node visited AFTER sol[b, i]
    prev_sol = torch.roll(sol, shifts=1, dims=1)
    next_sol = torch.roll(sol, shifts=-1, dims=1)

    # 3. Gather Distances using Advanced Indexing
    # We create a batch index tensor to select from the correct batch in the matrix
    # shape: [B, L]
    batch_idx = torch.arange(B, device=distance_matrix.device).unsqueeze(1).expand(B, L)

    # d_pi: Distance (Prev -> Curr)
    d_pi = distance_matrix[batch_idx, prev_sol, sol]

    # d_in: Distance (Curr -> Next)
    d_in = distance_matrix[batch_idx, sol, next_sol]

    # d_pn: Distance (Prev -> Next) - The "Shortcut"
    d_pn = distance_matrix[batch_idx, prev_sol, next_sol]

    # 4. Calculate Detour Cost for every position in the sequence
    # Shape: [B, L]
    detour_values = d_pi + d_in - d_pn

    # 5 Return with correct shape [B, N, 1]
    return detour_values.unsqueeze(-1)


def is_feasible(
    solution: torch.Tensor, demands: torch.Tensor, capacity: torch.Tensor
) -> torch.Tensor:
    """
    Check if the solution respects capacity constraints for all routes.

    Args:
        solution: Tensor [batch, route_length, 1] representing routes
        demands: Tensor [batch, route_length] with demands for each client (0 for depot)
        capacity: Tensor [batch] with vehicle capacities for each problem instance

    Returns:
        Tensor [batch] of boolean values indicating feasibility for each problem
    """
    batch_size = solution.size(0)
    device = solution.device

    mask = solution.squeeze(-1) != 0  # [batch, route_length]
    segment_start = mask & ~torch.cat(
        [torch.zeros_like(mask[:, :1]), mask[:, :-1]], dim=1
    )
    segment_ids = torch.cumsum(segment_start, 1) * mask
    num_routes = segment_ids.max() + 1

    route_loads = torch.zeros(
        batch_size,
        int(num_routes),
        device=device,
        dtype=demands.dtype,
    )
    route_loads.scatter_add_(1, segment_ids, demands)

    # Check if any route demand exceeds capacity
    feasible = (route_loads <= capacity).all(dim=1)
    start_w_depot = solution[:, 0, 0] == 0
    end_w_depot = solution[:, -1, 0] == 0
    feasible = feasible & start_w_depot & end_w_depot

    return feasible


def is_feasible2(
    solution: torch.Tensor, demands: torch.Tensor, capacity: torch.Tensor
) -> torch.Tensor:
    """
    Check if the solution respects capacity constraints for all routes.

    Args:
        solution: Tensor [batch, route_length, 1] representing routes
        demands: Tensor [batch, route_length] with demands for each client (0 for depot)
        capacity: Tensor [batch] with vehicle capacities for each problem instance

    Returns:
        Tensor [batch] of boolean values indicating feasibility for each problem
    """
    batch_size = solution.size(0)
    device = solution.device

    mask = solution.squeeze(-1) != 0  # [batch, route_length]

    # Identify route starts: True where a route starts (depot->client)
    route_starts = torch.cat(
        [
            torch.ones(batch_size, 1, dtype=torch.bool, device=device),
            (~mask[:, :-1]) & mask[:, 1:],
        ],
        dim=1,
    )  # [batch, route_length]

    # Assign a route id to each position
    route_ids = torch.cumsum(route_starts, dim=1) - 1  # [batch, route_length]

    # Set depot positions to -1 so they don't contribute to route sums
    route_ids[~mask] = -1  # depot positions

    # Compute route demand sums (scatter_add)
    max_routes = route_ids.max().item() + 1 if route_ids.numel() > 0 else 0
    max_routes = max(max_routes, solution.size(1))  # ensure enough space
    route_demands = torch.zeros(
        batch_size, int(max_routes), dtype=torch.int64, device=device
    )

    route_demands.scatter_add_(1, torch.clamp(route_ids, min=0), demands * mask)

    # Check if any route demand exceeds capacity
    feasible = (route_demands <= capacity).all(dim=1)

    start_w_depot = solution[:, 0, 0] == 0
    end_w_depot = solution[:, -1, 0] == 0
    feasible = feasible & start_w_depot & end_w_depot

    return feasible


def capacity_utilization(
    solution: torch.Tensor, demands: torch.Tensor, capacity: torch.Tensor
) -> torch.Tensor:
    """
    Calculate capacity utilization score for each solution.

    For each route in a solution, computes the ratio of total demand
    to vehicle capacity, then averages these ratios across all routes.
    Higher scores indicate better capacity utilization.

    Args:
        solution: Tensor [batch_size, route_length, 1] representing routes
        demands: Tensor [batch_size, route_length] with demands for each client
        (0 for depot)
        capacity: Tensor [batch_size] with vehicle capacities for each problem instance

    Returns:
        Tensor [batch_size] with utilization score for each solution
    """
    batch_size = solution.size(0)
    device = solution.device

    mask = solution.squeeze(-1) != 0  # [batch, route_length] - True for clients

    # Identify route starts: depot followed by client
    route_starts = torch.cat(
        [
            torch.ones(batch_size, 1, dtype=torch.bool, device=device),
            (~mask[:, :-1]) & mask[:, 1:],
        ],
        dim=1,
    )  # [batch, route_length]

    # Count number of routes per solution
    routes_count = route_starts.sum(dim=1).float()  # [batch_size]

    # Assign route IDs to each position
    route_ids = torch.cumsum(route_starts, dim=1) - 1  # [batch, route_length]

    # Mask depot positions so they don't contribute to demand sums
    route_ids[~mask] = -1

    # Compute total demand per route
    max_routes = route_ids.max().item() + 1 if route_ids.numel() > 0 else 0
    max_routes = max(max_routes, 1)  # ensure at least one route
    route_demands = torch.zeros(
        batch_size, int(max_routes), dtype=torch.float, device=device
    )

    route_demands.scatter_add_(
        1, torch.clamp(route_ids, min=0), demands.float() * mask.float()
    )

    # Calculate utilization ratio for each route (demand / capacity)
    route_utilization = route_demands / capacity

    # Get sum of utilization across all routes
    total_utilization = torch.sum(route_utilization, dim=1)

    # Compute average utilization per route
    avg_utilization = total_utilization / routes_count

    return 1 - avg_utilization


def find_indices(x, c1, c2):
    """
    Vectorized version for better efficiency.
    """
    # Ensure dimensions are compatible
    assert x.shape[0] == c1.shape[0] == c2.shape[0], (
        "The first dimension must be the same"
    )

    # Reshape tensors
    x_flat = x.squeeze(-1).long()  # [batch, pb_size]
    c1_flat = c1.view(-1, 1).long()  # [batch, 1]
    c2_flat = c2.view(-1, 1).long()  # [batch, 1]

    # Find indices using broadcasting
    # Compare each element of x with the corresponding value of c1/c2
    mask1 = x_flat == c1_flat
    mask2 = x_flat == c2_flat

    # Create a tensor of indices [0, 1, 2, ..., pb_size-1]
    indices_range = torch.arange(x_flat.shape[1], device=x.device).unsqueeze(0)

    # Use masks to retrieve indices
    # For values not found, set to -1
    idx1 = torch.where(
        mask1.any(dim=1),
        (mask1 * indices_range).max(dim=1)[0],
        torch.tensor(-1, device=x.device),
    )
    idx2 = torch.where(
        mask2.any(dim=1),
        (mask2 * indices_range).max(dim=1)[0],
        torch.tensor(-1, device=x.device),
    )

    # Combine results
    result = torch.stack([idx1, idx2], dim=1)

    return result


def get_values_from_indices(x, action):
    """
    Retrieve values c1 and c2 from indices in x.

    Args:
        x: Tensor of shape [batch_size, pb_size, 1]
        action: Tensor of shape [batch_size, 2] containing the indices

    Returns:
        c1: Tensor of shape [batch_size, 1]
        c2: Tensor of shape [batch_size, 1]
        Optional heuristic_idx: Tensor of shape [batch_size, 1] if action has 3 columns
    """
    x_flat = x.squeeze(-1)  # [batch_size, sequence_length]

    # Directly retrieve the values
    c1 = x_flat[torch.arange(x.shape[0]), action[:, 0].long()].unsqueeze(1).long()
    c2 = x_flat[torch.arange(x.shape[0]), action[:, 1].long()].unsqueeze(1).long()
    if action.shape[1] == 3:
        heuristic_idx = action[:, 2].unsqueeze(1).long()
        return c1, c2, heuristic_idx

    return c1, c2
