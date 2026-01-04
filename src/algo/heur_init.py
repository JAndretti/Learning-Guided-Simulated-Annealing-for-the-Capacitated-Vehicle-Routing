import collections
import multiprocessing as mp
import os

import torch
from tqdm import tqdm

# Padding multiplier for route tensors (60% extra space for depot visits)
MULT = 0.6


def generate_isolate_solution(cvrp_instance):
    """
    Generate initial CVRP solutions using the Isolate Algorithm.

    This creates a solution where each client is visited individually
    with depot returns between each visit: [0,1,0,2,0,3,0,4,...,0,N,0]

    Args:
        cvrp_instance: An instance of the CVRP class containing problem data

    Returns:
        torch.Tensor: Solution tensor of shape [batch_size, route_length, 1]
    """
    batch_size = cvrp_instance.n_problems
    dim = cvrp_instance.dim  # Number of clients (excluding depot)
    device = "cpu"

    # Pattern: [0, 1, 0, 2, ..., 0, N, 0]
    route_length = 2 * dim + 1  # Start depot, depot after each client, end depot

    # Build pattern for a single batch
    pattern = torch.zeros(route_length, dtype=torch.long, device=device)
    pattern[0] = 0  # Start at depot
    pattern[-1] = 0  # End at depot
    pattern[1:-1:2] = torch.arange(1, dim + 1, device=device)  # Clients
    pattern[2:-1:2] = 0  # Depot after each client

    # Repeat for batch
    routes = pattern.unsqueeze(0).repeat(batch_size, 1)
    return routes.unsqueeze(-1)


def generate_sweep_solution(cvrp_instance):
    """
    Generate initial CVRP solutions using the Sweep Algorithm.

    This classical heuristic constructs routes by:
    1. Sorting clients by polar angle around the depot
    2. Sweeping through clients in angular order
    3. Starting new routes when vehicle capacity would be exceeded

    Args:
        cvrp_instance: An instance of the CVRP class containing problem data

    Returns:
        torch.Tensor: Solution tensor of shape [batch_size, max_route_length, 1]
    """
    # --------------------------------
    # Extract problem parameters
    # --------------------------------
    batch_size = cvrp_instance.n_problems  # Number of problem instances in batch
    dim = cvrp_instance.dim  # Number of clients (excluding depot)
    num_total_nodes = dim + 1  # Total nodes including depot
    device = "cpu"  # Computation device (CPU/GPU)
    coords = cvrp_instance.coords.cpu()  # Node coordinates [batch, nodes, 2]
    demands = cvrp_instance.demands.cpu()  # Node demands [batch, nodes]
    capacity = cvrp_instance.capacity.squeeze(-1).cpu()  # Vehicle capacity constraint

    # --------------------------------
    # Initialize solution tensor
    # --------------------------------
    # Add padding for depot returns (60% extra space)
    max_route_len = num_total_nodes + int(num_total_nodes * MULT)
    routes = torch.zeros(batch_size, max_route_len, dtype=torch.long, device=device)

    # --------------------------------
    # Prepare coordinate calculations
    # --------------------------------
    # Extract depot coordinates (first node in each problem)
    depot_coords = coords[:, 0:1, :]  # [batch_size, 1, 2]
    # Extract client coordinates (all nodes except depot)
    client_coords = coords[:, 1:, :]  # [batch_size, dim, 2]

    # Calculate vectors from depot to each client
    delta_coords = client_coords - depot_coords  # [batch_size, dim, 2]

    # Calculate polar angles (in radians) for each client relative to depot
    # atan2(y, x) gives angle in radians in range [-π, π]
    angles = torch.atan2(
        delta_coords[:, :, 1], delta_coords[:, :, 0]
    )  # [batch_size, dim]

    # --------------------------------
    # Sort clients by polar angle
    # --------------------------------
    # Create client indices (1 to dim) for each problem
    client_indices = torch.arange(1, dim + 1, device=device)
    client_indices = client_indices.expand(batch_size, -1)  # [batch_size, dim]

    # Sort clients by their polar angles for each problem
    sorted_indices = []
    sorted_demands = []

    for b in range(batch_size):
        # Get sorted indices for this problem
        _, indices = torch.sort(angles[b])
        sorted_indices.append(client_indices[b, indices])
        # Get corresponding demands
        sorted_demands.append(demands[b, client_indices[b, indices]])

    # Stack tensors for batch processing
    sorted_indices = torch.stack(sorted_indices)  # [batch_size, dim]
    sorted_demands = torch.stack(sorted_demands)  # [batch_size, dim]

    # --------------------------------
    # Construct routes using sorted clients
    # --------------------------------
    # Initialize variables for route construction
    current_loads = torch.zeros(batch_size, device=device)
    route_pos = torch.ones(
        batch_size, dtype=torch.long, device=device
    )  # Start at pos 1 (after depot)

    # All routes start at depot
    routes[:, 0] = 0

    # Sweep through sorted clients
    for i in range(dim):
        client_ids = sorted_indices[:, i]
        client_demands = sorted_demands[:, i]

        # Check if adding client would exceed capacity
        capacity_check = current_loads + client_demands <= capacity

        # For problems where capacity would be exceeded:
        # 1. Return to depot (already filled with 0s)
        # 2. Reset current load to 0
        route_pos = torch.where(capacity_check, route_pos, route_pos + 1)
        current_loads = torch.where(
            capacity_check, current_loads + client_demands, client_demands
        )

        # Add client to route
        for b in range(batch_size):
            if route_pos[b] < max_route_len:
                routes[b, route_pos[b]] = client_ids[b]

        # Move position pointer
        route_pos += 1

        # Check for route_pos overflow
        if (route_pos >= max_route_len).any():
            print("Warning: Some routes may be truncated")

    # Return to depot at the end of each route if not already there
    for b in range(batch_size):
        if route_pos[b] < max_route_len and routes[b, route_pos[b] - 1] != 0:
            routes[b, route_pos[b]] = 0

    # Add final dimension to match expected shape
    return routes.unsqueeze(-1)


def random_init_batch(cvrp_instance) -> torch.Tensor:
    """
    Vectorized random algorithm for initial CVRP solution generation.

    Args:
        demands: Tensor of shape [batch_size, num_nodes] representing client demands
        capacity: Vehicle capacity constraint

    Returns:
        Tensor of shape [batch_size, max_route_length, 1] containing initial routes
        with depot visits inserted to respect capacity constraints
    """
    demands = cvrp_instance.demands  # Ensure demands are on CPU
    capacity = cvrp_instance.capacity.squeeze(-1)  # Ensure capacity is a scalar
    batch_size, num_nodes = demands.size()
    device = demands.device

    # Initialize solution tensor with 40% padding for depot insertions

    # Prepare client indices (excluding depot 0)
    clients = torch.arange(1, num_nodes, device=device)
    clients = clients.repeat(batch_size, 1)  # [batch_size, num_nodes-1]

    # Shuffle clients differently for each problem in batch
    rand_floats = torch.rand_like(clients, dtype=torch.float)
    indices = torch.argsort(rand_floats, dim=1)
    clients = torch.gather(clients, 1, indices)

    return construct_cvrp_solution(clients, demands, capacity)

    # # Add depot (0) at start of each route
    # clients = torch.cat(
    #     [
    #         torch.zeros(clients.size(0), 1, dtype=clients.dtype, device=device),
    #         clients,
    #     ],
    #     dim=1,
    # )

    # # Initialize tracking variables
    # remaining_capacity = capacity
    # route_positions = torch.zeros(batch_size, dtype=torch.long, device=device)
    # vehicle_count = torch.ones(batch_size, device=device)

    # # Build routes by sequentially assigning clients
    # while (clients >= 0).any():  # While clients remain unassigned
    #     current_client = clients[:, 0]

    #     # Check if client demand can be satisfied with current capacity
    #     can_serve = (
    #         demands[torch.arange(batch_size), current_client] <= remaining_capacity
    #     )

    #     # Update route with client or depot (0)
    #     routes[torch.arange(batch_size), route_positions] = torch.where(
    #         can_serve, current_client, 0
    #     )
    #     route_positions += 1

    #     # Update remaining capacity
    #     remaining_capacity = torch.where(
    #         can_serve,
    #         remaining_capacity - demands[torch.arange(batch_size), current_client],
    #         remaining_capacity,
    #     )

    #     # If can't serve, return to depot and dispatch new vehicle
    #     remaining_capacity = torch.where(can_serve, remaining_capacity, capacity)
    #     vehicle_count = torch.where(can_serve, vehicle_count, vehicle_count + 1)

    #     # Remove assigned client from consideration
    #     clients = torch.where(
    #         can_serve.unsqueeze(1),
    #         torch.cat(
    #             [
    #                 clients[:, 1:],
    #                 torch.full((batch_size, 1), -1, device=clients.device),
    #             ],
    #             dim=1,
    #         ),
    #         clients,
    #     )

    # # Ensure all negative values (padding) are set to depot (0)
    # routes = torch.where(routes < 0, torch.tensor(0, device=routes.device), routes)

    # return routes.unsqueeze(-1)  # Add dimension for compatibility


def _vrp_optimal_split_worker(args):
    """
    Worker function for parallel processing of VRP optimal split algorithm.

    Args:
        args: Tuple containing (batch_index, coordinates, demands, max_loads,
        client_order)

    Returns:
        Tuple of (batch_index, solution_tensor)
    """
    batch_index, coordinates, demands, max_loads, client_order = args
    device = "cpu"

    # Extract data for this single instance
    n_clients = client_order.shape[0]

    if n_clients == 0:
        return batch_index, torch.zeros(1, dtype=torch.long, device=device)

    # --- Precomputations ---

    # Full Euclidean distance matrix for this instance
    coords_i = coordinates.unsqueeze(1)
    coords_j = coordinates.unsqueeze(0)
    dist = torch.norm(coords_i - coords_j, dim=2)

    # Retrieve demands and coordinates in the order of the tour
    ordered_demands = demands[client_order]  # [n_clients, 1]

    # Feasibility check
    if (ordered_demands.squeeze(-1) > max_loads).any():
        raise ValueError(
            f"Problem {batch_index} infeasible: some clients have demands exceeding "
            "capacity."
        )

    # Cumulative calculations for the O(n) algorithm
    # Q_cum[i] = cumulative demand up to the i-th client in the tour
    Q_cum = torch.cumsum(ordered_demands.squeeze(-1), dim=0)

    # D_cum[i] = cumulative distance along the tour up to the i-th client
    # Compute distances between consecutive clients in the order
    if n_clients > 1:
        ordered_coords_prev = coordinates[client_order[:-1]]
        ordered_coords_next = coordinates[client_order[1:]]
        inter_client_dist = torch.norm(ordered_coords_next - ordered_coords_prev, dim=1)
        D_cum = torch.cumsum(inter_client_dist, dim=0)
        # Add a 0 at the beginning so that D_cum[i] is the distance up to client i
        D_cum = torch.cat([torch.zeros(1, device=device), D_cum], dim=0)
    else:
        D_cum = torch.zeros(1, device=device)

    # --- O(n) algorithm ---

    max_load = max_loads.item()

    # p[i] = cost of the best path from the depot to the i-th client in the tour
    p = torch.full((n_clients + 1,), float("inf"), device=device)
    # pred[i] = optimal predecessor (split point) for client i
    pred = torch.zeros((n_clients + 1,), dtype=torch.long, device=device)

    p[0] = 0
    # The deque contains indices of potential predecessors (0 to n)
    deque = collections.deque([0])

    # g_vals[i] = fixed cost associated with predecessor i
    # g_i = p[i] + dist(depot, C_i+1) - D_cum[i]
    # Compute distances from the depot to clients once
    dist_depot_to_ordered_clients = dist[0, client_order]

    for t in range(1, n_clients + 1):
        # Remove predecessors from the deque that no longer respect capacity
        # to serve client t (at index t-1 in 0-indexed arrays)
        while (
            deque
            and Q_cum[t - 1] - (Q_cum[deque[0] - 1] if deque[0] > 0 else 0) > max_load
        ):
            deque.popleft()

        if not deque:
            raise ValueError(
                f"Problem {batch_index} infeasible, no valid predecessor found."
            )

        # The best predecessor is always the first in the deque
        best_pred_idx = deque[0]

        # Cost of the new route: Depot -> C_{i+1} -> ... -> C_t -> Depot
        # (i is best_pred_idx, t is the current index)
        cost_depot_to_first = (
            dist_depot_to_ordered_clients[best_pred_idx]
            if best_pred_idx < n_clients
            else dist[0, client_order[-1]]
        )
        dist_in_tour = D_cum[t - 1] - D_cum[best_pred_idx]
        cost_last_to_depot = dist[client_order[t - 1], 0]

        cost_route = cost_depot_to_first + dist_in_tour + cost_last_to_depot

        p[t] = p[best_pred_idx] + cost_route
        pred[t] = best_pred_idx

        # Update the deque with t as a new potential predecessor
        if t < n_clients:
            # Compute the 'g' value for the new predecessor t
            # g_t = p[t] + dist(depot, C_t+1) - D_cum[t]
            g_t = p[t] + dist_depot_to_ordered_clients[t] - D_cum[t]

            # Remove predecessors dominated by t from the deque
            while deque:
                j = deque[-1]
                # Compute g_j
                dist_depot_j = (
                    dist_depot_to_ordered_clients[j]
                    if j < n_clients
                    else dist[0, client_order[-1]]
                )
                g_j = p[j] + dist_depot_j - D_cum[j]
                if g_t <= g_j:
                    deque.pop()
                else:
                    break
            deque.append(t)

    # --- Backtracking ---
    solution = [0]
    split_points = []
    curr = n_clients
    while curr > 0:
        split_points.append(curr)
        curr = pred[curr].item()

    # Reconstruct the solution from the split points
    tour_segment = []
    for i in range(n_clients):
        tour_segment.append(client_order[i].item())
        if (i + 1) in split_points:
            solution.extend(tour_segment)
            solution.append(0)
            tour_segment = []

    return batch_index, torch.tensor(solution, dtype=torch.long, device=device)


def vrp_optimal_split(coordinates, demands, max_loads, client_order):
    """
    Parallelized version of VRP optimal split algorithm.

    This function implements the linear-time algorithm by Vidal (2016)
    to find the optimal split of a client tour, parallelized across CPU cores.

    Args:
        coordinates: Tensor [batch_size, n_clients+1, 2] - coordinates
                    (depot at position 0)
        demands: Tensor [batch_size, n_clients+1, 1] - demands (depot = 0)
        max_loads: Tensor [batch_size, 1] - maximum capacity per problem
        client_order: Tensor [batch_size, n_clients] - client order (excluding depot)

    Returns:
        solutions: Tensor [batch_size, max_solution_length, 1] -
                  solutions with depots inserted
    """
    batch_size = coordinates.shape[0]
    n_clients_total = coordinates.shape[1] - 1
    device = coordinates.device

    if n_clients_total == 0:
        return torch.zeros(batch_size, 1, 1, dtype=torch.long, device=device)

    # Move tensors to CPU for multiprocessing
    coordinates_cpu = coordinates.cpu()
    demands_cpu = demands.cpu()
    max_loads_cpu = max_loads.cpu()
    client_order_cpu = client_order.cpu()

    # Determine number of processes (use all available CPU cores)
    num_processes = min(batch_size, os.cpu_count())

    # Prepare arguments for parallel processing
    args_list = []
    for b in range(batch_size):
        args_list.append(
            (
                b,
                coordinates_cpu[b],
                demands_cpu[b],
                max_loads_cpu[b],
                client_order_cpu[b],
            )
        )

    # Process in parallel
    if num_processes > 1 and batch_size > 1:
        with mp.Pool(processes=num_processes) as pool:
            results = list(
                tqdm(
                    pool.imap(_vrp_optimal_split_worker, args_list),
                    total=batch_size,
                    desc="VRP Optimal Split",
                    leave=False,
                )
            )
    else:
        # Sequential processing for small batches or single core
        results = []
        for args in tqdm(args_list, desc="VRP Optimal Split"):
            results.append(_vrp_optimal_split_worker(args))

    # Sort results by batch index to maintain order
    results.sort(key=lambda x: x[0])

    # Collect solutions and pad to uniform length
    solutions = [result[1] for result in results]
    max_len = max(len(sol) for sol in solutions) if solutions else 1

    padded_solutions = torch.zeros(
        batch_size, max_len, 1, dtype=torch.long, device=device
    )

    for i, sol in enumerate(solutions):
        padded_solutions[i, : len(sol), 0] = sol.to(device)

    return padded_solutions


def construct_cvrp_solution(
    x: torch.Tensor, demands: torch.Tensor, capacity: torch.Tensor
) -> torch.Tensor:
    """
    Constructs a valid CVRP solution from node sequence while respecting
    capacity constraints.

    Args:
        x: Tensor of shape [batch_size, num_nodes] representing node visit sequence
        demands: Tensor of shape [batch_size, num_nodes+1] containing client demands
        capacity: Maximum vehicle capacity

    Returns:
        Tensor of shape [batch_size, max_route_length, 1] with depot visits inserted
        to maintain feasible solutions
    """
    x = x.squeeze(-1)  # Remove trailing dimension if present
    batch_size, num_nodes = demands.shape
    device = x.device

    # Initialize solution with padding for depot insertions
    max_route_length = num_nodes + int(num_nodes * MULT)
    routes = torch.zeros(batch_size, max_route_length, dtype=torch.long, device=device)

    # Add depot at start of each route
    x = torch.cat(
        [torch.zeros(batch_size, 1, dtype=torch.long, device=device), x], dim=1
    )

    # Get demands in current route order
    ordered_demands = torch.gather(demands, 1, x)

    # Initialize tracking variables
    capacity = capacity.squeeze()
    remaining_capacity = capacity.long()
    route_pos = torch.zeros(batch_size, dtype=torch.long, device=device)

    for i in range(num_nodes):
        current_client = x[:, i]
        can_serve = ordered_demands[:, i] <= remaining_capacity

        # Insert client or depot based on capacity
        routes[torch.arange(batch_size), route_pos] = torch.where(
            can_serve, current_client, 0
        )
        route_pos += 1

        # Update capacity
        remaining_capacity = torch.where(
            can_serve, remaining_capacity - ordered_demands[:, i], remaining_capacity
        )

        # If capacity exceeded, return to depot and reset
        remaining_capacity = torch.where(
            can_serve, remaining_capacity, capacity - ordered_demands[:, i]
        )

        # Insert depot if needed
        routes[torch.arange(batch_size), route_pos] = torch.where(
            can_serve, 0, current_client
        )
        route_pos += (~can_serve).long()

    return routes.unsqueeze(-1)


def generate_nearest_neighbor(cvrp_instance):
    """
    Generate initial CVRP solutions using the Nearest Neighbor Heuristic.

    This heuristic starts at the depot and iteratively visits the closest unvisited
    customer,
    returning to the depot when vehicle capacity would be exceeded.

    Args:
        cvrp_instance: An instance of the CVRP class containing problem data

    Returns:
        torch.Tensor: Solution tensor of shape [batch_size, max_route_length, 1]
    """
    # Extract problem parameters
    batch_size = cvrp_instance.n_problems
    dim = cvrp_instance.dim  # Number of clients (excluding depot)
    device = "cpu"
    coords = cvrp_instance.coords.cpu()
    demands = cvrp_instance.demands.cpu()
    capacity = cvrp_instance.capacity.squeeze(-1).cpu()

    # Compute distance matrix
    dist_matrix = torch.cdist(coords, coords, p=2)

    # Initialize routes tensor with padding
    max_route_len = dim + 1 + int((dim + 1) * MULT)
    routes = torch.zeros(batch_size, max_route_len, dtype=torch.long, device=device)

    # Track visited clients (True for visited)
    visited = torch.zeros(batch_size, dim + 1, dtype=torch.bool, device=device)
    visited[:, 0] = True  # Depot is not a client to be visited

    # Initialize state for each problem in the batch
    current_node = torch.zeros(batch_size, dtype=torch.long, device=device)
    current_load = torch.zeros(batch_size, device=device)
    route_pos = torch.ones(batch_size, dtype=torch.long, device=device)

    # All routes start at the depot
    routes[:, 0] = 0

    # Loop until all clients are visited
    while not visited[:, 1:].all():
        # Create a mask for unvisited clients
        unvisited_mask = ~visited

        # Set distances to visited nodes to infinity
        dist_from_current = dist_matrix[torch.arange(batch_size), current_node].clone()
        dist_from_current[unvisited_mask] = dist_from_current[unvisited_mask]
        dist_from_current[~unvisited_mask] = float("inf")
        dist_from_current[:, 0] = float("inf")  # Don't go back to depot unless needed

        # Find the nearest unvisited neighbor for each problem
        nearest_neighbor = torch.argmin(dist_from_current, dim=1)

        # For nodes where all clients are visited, nearest_neighbor might be invalid
        # Set to depot for those cases
        all_clients_visited = visited[:, 1:].all(dim=1)
        nearest_neighbor = torch.where(
            all_clients_visited, torch.zeros_like(nearest_neighbor), nearest_neighbor
        )

        # Get demands of nearest neighbors
        neighbor_demands = demands[torch.arange(batch_size), nearest_neighbor]

        # Check if adding the neighbor exceeds capacity
        over_capacity = (current_load + neighbor_demands) > capacity

        # For problems where capacity is exceeded or all clients visited:
        # 1. Return to depot (add 0 to route)
        # 2. Reset load
        # 3. Update current node to depot
        routes[torch.arange(batch_size), route_pos] = torch.where(
            over_capacity | all_clients_visited,
            torch.zeros_like(nearest_neighbor),
            nearest_neighbor,
        )

        # Update route position for all problems
        route_pos += 1

        # For problems that returned to depot, add another position
        current_node = torch.where(
            over_capacity | all_clients_visited,
            torch.zeros_like(current_node),  # Return to depot
            nearest_neighbor,  # Go to nearest neighbor
        )

        # Update current load based on action taken
        current_load = torch.where(
            over_capacity | all_clients_visited,
            torch.zeros_like(current_load),  # Reset if returned to depot
            current_load + neighbor_demands,  # Add demand if visited client
        )

        # Mark nearest neighbors as visited (except depot)
        for b in range(batch_size):
            if not all_clients_visited[b] and not over_capacity[b]:
                visited[b, nearest_neighbor[b]] = True

    # Ensure final return to depot
    for b in range(batch_size):
        if route_pos[b] < max_route_len and routes[b, route_pos[b] - 1] != 0:
            routes[b, route_pos[b]] = 0
            route_pos[b] += 1

    return routes.unsqueeze(-1)


def _clark_wright_worker(args):
    """
    Worker function for parallel processing of Clarke and Wright algorithm.

    Args:
        args: Tuple containing (batch_index, coords, demands, capacity, dim)

    Returns:
        Tuple of (batch_index, flattened_route)
    """
    batch_index, coords, demands, capacity, dim = args
    device = "cpu"

    # Compute distance matrix for single batch
    coords_i = coords.unsqueeze(1)
    coords_j = coords.unsqueeze(0)
    dist = torch.norm(coords_i - coords_j, dim=2)

    # Compute savings matrix
    d0i = dist[0, 1:]  # [dim]
    d0j = dist[0, 1:]  # [dim]
    dij = dist[1:, 1:]  # [dim, dim]
    savings = d0i.unsqueeze(1) + d0j.unsqueeze(0) - dij  # [dim, dim]
    # Set diagonal to -inf to avoid merging a node with itself
    savings = savings + torch.diag(torch.full((dim,), float("-inf"), device=device))

    # Get sorted savings indices (descending)
    savings_flat = savings.view(-1)
    sorted_savings, sorted_idx = torch.sort(savings_flat, descending=True)
    # Map flat indices back to (i, j)
    i_idx = sorted_idx // dim
    j_idx = sorted_idx % dim
    # Convert to global node indices (clients are 1..dim)
    i_idx = i_idx + 1
    j_idx = j_idx + 1

    # Initialize routes: each client in its own route [0, i, 0]
    routes = [[0, i + 1, 0] for i in range(dim)]
    route_demands = [demands[i + 1].item() for i in range(dim)]
    # Track which route each client is in
    client_route = {i + 1: i for i in range(dim)}

    # Try to merge routes according to savings
    for k in range(dim * dim):
        i = i_idx[k].item()
        j = j_idx[k].item()
        if i == j:
            continue
        route_i = client_route.get(i, None)
        route_j = client_route.get(j, None)
        if route_i is None or route_j is None or route_i == route_j:
            continue
        # Check if i is at the end of its route and j at the start of its route
        route_i_seq = routes[route_i]
        route_j_seq = routes[route_j]
        if route_i_seq[-2] == i and route_j_seq[1] == j:
            total_demand = route_demands[route_i] + route_demands[route_j]
            if total_demand <= capacity.item():
                # Merge route_i and route_j
                new_route = route_i_seq[:-1] + route_j_seq[1:]
                routes[route_i] = new_route
                route_demands[route_i] = total_demand
                # Remove route_j
                routes[route_j] = []
                route_demands[route_j] = 0
                # Update client_route mapping
                for node in route_j_seq[1:-1]:
                    client_route[node] = route_i
                continue
        # Also check the reverse: j at end, i at start
        if route_j_seq[-2] == j and route_i_seq[1] == i:
            total_demand = route_demands[route_i] + route_demands[route_j]
            if total_demand <= capacity.item():
                new_route = route_j_seq[:-1] + route_i_seq[1:]
                routes[route_j] = new_route
                route_demands[route_j] = total_demand
                routes[route_i] = []
                route_demands[route_i] = 0
                for node in route_i_seq[1:-1]:
                    client_route[node] = route_j
                continue

    # Collect non-empty routes and flatten
    flat = []
    for r in routes:
        if r:
            if flat and flat[-1] != 0:
                flat.append(0)
            flat += r[1:] if flat else r
    # Ensure route starts at depot
    if not flat or flat[0] != 0:
        flat = [0] + flat

    return batch_index, flat


def generate_Clark_and_Wright(cvrp_instance):
    """
    Generate initial CVRP solutions using the Clarke and Wright Savings Algorithm
    (Parallelized).

    This algorithm works by:
    1. Initially creating N separate routes (depot-client-depot) for each client
    2. Computing savings from merging pairs of routes
    3. Iteratively merging routes with highest savings that maintain feasibility

    Args:
        cvrp_instance: An instance of the CVRP class containing problem data

    Returns:
        torch.Tensor: Solution tensor of shape [batch_size, max_route_length, 1]
    """
    batch_size = cvrp_instance.n_problems
    dim = cvrp_instance.dim  # Number of clients (excluding depot)
    num_total_nodes = dim + 1
    device = "cpu"
    coords = cvrp_instance.coords.cpu()
    demands = cvrp_instance.demands.cpu()
    capacity = cvrp_instance.capacity.squeeze(-1).cpu()

    # Determine number of processes (use all available CPU cores)
    num_processes = min(batch_size, os.cpu_count())

    # Prepare arguments for parallel processing
    args_list = []
    for b in range(batch_size):
        args_list.append((b, coords[b], demands[b], capacity[b], dim))

    # Process in parallel
    if num_processes > 1 and batch_size > 1:
        with mp.Pool(processes=num_processes) as pool:
            results = list(
                tqdm(
                    pool.imap(_clark_wright_worker, args_list),
                    total=batch_size,
                    desc="Clark and Wright Init",
                    leave=False,
                )
            )
    else:
        # Sequential processing for small batches or single core
        results = []
        for args in tqdm(args_list, desc="Clark and Wright Init"):
            results.append(_clark_wright_worker(args))

    # Sort results by batch index to maintain order
    results.sort(key=lambda x: x[0])

    # Collect routes and pad to uniform length
    max_route_len = num_total_nodes + int(num_total_nodes * MULT)
    batch_routes = torch.zeros(
        batch_size, max_route_len, dtype=torch.long, device=device
    )

    for batch_idx, flat_route in results:
        # Pad to max_route_len
        flat_route = flat_route[:max_route_len] + [0] * (
            max_route_len - len(flat_route)
        )
        batch_routes[batch_idx, : len(flat_route)] = torch.tensor(
            flat_route[:max_route_len], dtype=torch.long
        )

    return batch_routes.unsqueeze(-1)


def cheapest_insertion(cvrp_instance):
    """
    Vectorized implementation of the Cheapest Insertion algorithm for the CVRP.

    Args:
        cvrp_instance: An instance of the CVRP class containing problem data

    Returns:
        routes: tensor of shape [batch_size, solution_length] containing the routes
    """

    batch_size = cvrp_instance.n_problems
    n_clients = cvrp_instance.dim
    n_total = n_clients + 1  # Including depot
    distance_matrix = cvrp_instance.matrix.cpu()
    demand = cvrp_instance.demands.cpu()
    vehicle_capacity = cvrp_instance.capacity.squeeze(-1).cpu()
    device = "cpu"

    # Initialization: for each problem, start with an empty route [0, 0]
    routes = [
        torch.tensor([0, 0], dtype=torch.long, device=device) for _ in range(batch_size)
    ]
    remaining_clients = (
        torch.arange(1, n_clients + 1, device=device)
        .unsqueeze(0)
        .expand(batch_size, -1)
    )
    current_load = torch.zeros(batch_size, device=device)

    # Mask to track which clients have been inserted
    inserted = torch.zeros((batch_size, n_total), dtype=torch.bool, device=device)
    inserted[:, 0] = True  # The depot is always inserted

    while True:
        # Find clients not yet inserted
        not_inserted = ~inserted
        not_inserted[:, 0] = False  # Ignore the depot

        # If all clients are inserted, we are done
        if not_inserted.sum().item() == 0:
            break

        # For each problem in the batch
        for b in range(batch_size):
            if not_inserted[b].sum().item() == 0:
                continue  # All clients are already inserted for this problem

            best_cost = float("inf")
            best_position = -1
            best_client = -1

            # Find all possible positions for insertion
            depot_positions = (routes[b] == 0).nonzero().flatten()

            for i in range(len(depot_positions) - 1):
                start_idx = depot_positions[i]
                end_idx = depot_positions[i + 1]
                subroute = routes[b][start_idx : end_idx + 1]

                # Calculate the current demand of this subroute
                route_demand = (
                    demand[b, subroute[1:-1]].sum() if len(subroute) > 2 else 0
                )
                remaining_capacity = vehicle_capacity[b] - route_demand

                # Find eligible clients for this route
                eligible_clients = remaining_clients[b][
                    (demand[b, remaining_clients[b]] <= remaining_capacity)
                    & not_inserted[b, remaining_clients[b]]
                ]

                if len(eligible_clients) == 0:
                    continue

                # For each pair of consecutive nodes in the subroute
                for k in range(len(subroute) - 1):
                    u = subroute[k]
                    v = subroute[k + 1]

                    # Calculate the insertion cost for each eligible client
                    delta = (
                        distance_matrix[b, u, eligible_clients]
                        + distance_matrix[b, eligible_clients, v]
                        - distance_matrix[b, u, v]
                    )

                    # Find the best client for this insertion
                    min_delta, min_idx = delta.min(dim=0)
                    if min_delta < best_cost:
                        best_cost = min_delta.item()
                        best_client = eligible_clients[min_idx].item()
                        best_position = start_idx + k + 1

            # If no insertion is possible in existing routes, create a new route
            if best_cost == float("inf"):
                # Find the closest uninserted client to the depot
                eligible = remaining_clients[b][not_inserted[b, remaining_clients[b]]]
                if len(eligible) == 0:
                    continue

                depot_dist = distance_matrix[b, 0, eligible]
                _, closest_idx = depot_dist.min(dim=0)
                best_client = eligible[closest_idx].item()

                # Find the position after the last 0
                last_zero_pos = (routes[b] == 0).nonzero()[-1].item()
                routes[b] = torch.cat(
                    [
                        routes[b][: last_zero_pos + 1],
                        torch.tensor([best_client, 0], device=device),
                        routes[b][last_zero_pos + 1 :],
                    ]
                )
                inserted[b, best_client] = True
                current_load[b] = demand[b, best_client]
            else:
                # Insert the best client found
                routes[b] = torch.cat(
                    [
                        routes[b][:best_position],
                        torch.tensor([best_client], device=device),
                        routes[b][best_position:],
                    ]
                )
                inserted[b, best_client] = True
                current_load[b] += demand[b, best_client]

    # Convert the list of routes into a single tensor with padding if necessary
    max_len = max(len(r) for r in routes) + int(n_clients * MULT)
    padded_routes = torch.zeros((batch_size, max_len), dtype=torch.long, device=device)
    for b in range(batch_size):
        padded_routes[b, : len(routes[b])] = routes[b]

    return padded_routes.unsqueeze(-1)


def path_cheapest_arc(cvrp_instance):
    """
    Implementation of the Path Cheapest Arc for the CVRP with strict constraints.

    Args:
        cvrp_instance: An instance of the CVRP class containing problem data

    Returns:
        routes: tensor [batch_size, max_len] (sequence of type [0,1,2,0,3,4,0,...])
    """
    batch_size = cvrp_instance.n_problems
    n_clients = cvrp_instance.dim
    n_total = n_clients + 1  # Including depot
    distance_matrix = cvrp_instance.matrix.cpu()
    demand = cvrp_instance.demands.cpu()
    vehicle_capacity = cvrp_instance.capacity.squeeze(-1).cpu()
    device = "cpu"

    # Initialization
    routes = [
        torch.tensor([0], dtype=torch.long, device=device) for _ in range(batch_size)
    ]
    remaining_clients = (
        torch.arange(1, n_clients + 1, device=device)
        .unsqueeze(0)
        .expand(batch_size, -1)
    )
    inserted = torch.zeros((batch_size, n_total), dtype=torch.bool, device=device)
    inserted[:, 0] = True

    while True:
        # Check if all clients are inserted
        not_inserted = ~inserted
        not_inserted[:, 0] = False  # Ignore the depot
        if not_inserted.sum() == 0:
            break

        for b in range(batch_size):
            if not_inserted[b].sum() == 0:
                continue

            best_cost = float("inf")
            best_client = -1
            best_pos = -1
            new_route_needed = True

            # Calculate the used capacity in each existing sub-route
            route = routes[b]
            depot_positions = (route == 0).nonzero().flatten()

            for i in range(len(depot_positions) - 1):
                start, end = depot_positions[i], depot_positions[i + 1]
                subroute = route[start : end + 1]

                # Calculate the current demand
                current_demand = (
                    demand[b, subroute[1:-1]].sum() if len(subroute) > 2 else 0
                )
                remaining_cap = vehicle_capacity[b] - current_demand

                # Eligible clients for this sub-route
                eligible = remaining_clients[b][
                    (demand[b, remaining_clients[b]] <= remaining_cap)
                    & not_inserted[b, remaining_clients[b]]
                ]

                if len(eligible) == 0:
                    continue

                # Find the best arc for insertion
                for j in range(len(subroute) - 1):
                    u, v = subroute[j], subroute[j + 1]
                    delta = (
                        distance_matrix[b, u, eligible]
                        + distance_matrix[b, eligible, v]
                        - distance_matrix[b, u, v]
                    )
                    min_delta, min_idx = delta.min(dim=0)

                    if min_delta < best_cost:
                        best_cost = min_delta.item()
                        best_client = eligible[min_idx].item()
                        best_pos = start + j + 1
                        new_route_needed = False

            # Handle new routes if necessary
            if new_route_needed:
                eligible = remaining_clients[b][not_inserted[b, remaining_clients[b]]]
                if len(eligible) == 0:
                    continue

                # Choose the closest client to the depot
                depot_dists = distance_matrix[b, 0, eligible]
                _, closest_idx = depot_dists.min(dim=0)
                best_client = eligible[closest_idx].item()

                # Add new route [0, client, 0] at the end
                if routes[b][-1] != 0:
                    routes[b] = torch.cat([routes[b], torch.tensor([0], device=device)])
                routes[b] = torch.cat(
                    [routes[b], torch.tensor([best_client, 0], device=device)]
                )
            else:
                # Insert the client at the optimal position
                routes[b] = torch.cat(
                    [
                        routes[b][:best_pos],
                        torch.tensor([best_client], device=device),
                        routes[b][best_pos:],
                    ]
                )

            inserted[b, best_client] = True

    # Padding for uniform output format
    max_len = max(len(r) for r in routes) + int(n_clients * MULT)
    padded_routes = torch.zeros((batch_size, max_len), dtype=torch.long, device=device)
    for b in range(batch_size):
        padded_routes[b, : len(routes[b])] = routes[b]

    return padded_routes.unsqueeze(-1)  # Add dimension for compatibility


def farthest_insertion(cvrp_instance):
    """
    Generate initial CVRP solutions using the optimized Farthest Insertion algorithm.

    Key improvements:
    1. Vectorized distance calculations
    2. Python lists for efficient route building
    3. Early termination when all clients are inserted
    4. Clearer documentation and comments

    Args:
        cvrp_instance: An instance of the CVRP class containing problem data

    Returns:
        torch.Tensor: Solution tensor of shape [batch_size, max_route_length, 1]
    """
    batch_size = cvrp_instance.n_problems
    n_clients = cvrp_instance.dim
    n_total = n_clients + 1  # Including depot
    distance_matrix = cvrp_instance.matrix.cpu()
    demand = cvrp_instance.demands.cpu()
    vehicle_capacity = cvrp_instance.capacity.squeeze(-1).cpu()
    device = "cpu"

    # Initialize routes as lists for efficient insertion
    routes = [[0] for _ in range(batch_size)]  # Each route starts at depot
    inserted = torch.zeros((batch_size, n_total), dtype=torch.bool, device=device)
    inserted[:, 0] = True  # Depot is always inserted

    for b in tqdm(range(batch_size), desc="Farthest Insertion INIT", leave=False):
        # Step 1: Start with client farthest from depot
        depot_distances = distance_matrix[b, 0, 1:]  # Shape: [n_clients]
        farthest_client = torch.argmax(depot_distances).item() + 1

        # Initialize first route: [0, farthest_client, 0]
        routes[b] = [0, farthest_client, 0]
        inserted[b, farthest_client] = True

        # Step 2: Insert remaining clients
        while not inserted[b, 1:].all():
            # Find all unvisited clients (indices 1..n_clients)
            unvisited = torch.where(~inserted[b, 1:])[0] + 1

            # Early exit if all clients are inserted
            if len(unvisited) == 0:
                break

            # Vectorized computation of farthest unvisited client
            # dist_matrix shape: [n_unvisited, n_visited]
            dist_to_visited = distance_matrix[b, unvisited][:, inserted[b]]
            min_distances = dist_to_visited.min(dim=1).values
            farthest_unvisited = unvisited[min_distances.argmax()].item()

            # Step 3: Find best insertion position
            best_cost = float("inf")
            best_position = -1
            can_insert = False

            # Split current route into subroutes between depots
            depot_positions = [i for i, x in enumerate(routes[b]) if x == 0]

            for i in range(len(depot_positions) - 1):
                start, end = depot_positions[i], depot_positions[i + 1]
                subroute = routes[b][start : end + 1]

                # Calculate current demand of subroute (excluding depots)
                current_demand = sum(demand[b, c] for c in subroute[1:-1])

                # Check capacity constraint
                if (
                    current_demand + demand[b, farthest_unvisited]
                    <= vehicle_capacity[b]
                ):
                    # Evaluate all possible insertion positions
                    for j in range(len(subroute) - 1):
                        u, v = subroute[j], subroute[j + 1]
                        cost = (
                            distance_matrix[b, u, farthest_unvisited]
                            + distance_matrix[b, farthest_unvisited, v]
                            - distance_matrix[b, u, v]
                        )

                        if cost < best_cost:
                            best_cost = cost
                            best_position = start + j + 1
                            can_insert = True

            # Step 4: Insert the client
            if can_insert:
                routes[b].insert(best_position, farthest_unvisited)
            else:
                # Create new route if no feasible insertion found
                routes[b].extend([farthest_unvisited, 0])

            inserted[b, farthest_unvisited] = True

        # Convert list route to tensor
        routes[b] = torch.tensor(routes[b], dtype=torch.long, device=device)

    # Convert to padded tensor format
    max_len = max(len(r) for r in routes) + int(n_clients * MULT)
    padded_routes = torch.zeros((batch_size, max_len), dtype=torch.long, device=device)

    for b in range(batch_size):
        padded_routes[b, : len(routes[b])] = routes[b]

    return padded_routes.unsqueeze(-1)
