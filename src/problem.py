# ============================================================================
# PROBLEM DEFINITION: CAPACITATED VEHICLE ROUTING PROBLEM (CVRP)
# ============================================================================

from abc import ABC, abstractmethod

import torch
from functools import partial
import torch.nn.functional as F

# Local Imports
from algo import (
    cheapest_insertion,
    construct_cvrp_solution,
    farthest_insertion,
    generate_Clark_and_Wright,
    generate_isolate_solution,
    generate_nearest_neighbor,
    generate_sweep_solution,
    insertion,
    path_cheapest_arc,
    random_init_batch,
    swap,
    two_opt,
    two_opt_star,
)
from utils import (
    calculate_client_angles,
    calculate_detour_features,
    calculate_distance_matrix,
    calculate_knn_isolation,
    calculate_neighbor_rank_features,
    is_feasible,
    repeat_to,
)

# Registry of initialization methods
INIT_METHODS = {
    "random": random_init_batch,
    "sweep": generate_sweep_solution,
    "isolate": generate_isolate_solution,
    "Clark_and_Wright": generate_Clark_and_Wright,
    "Clark_and_Wright_reversal": partial(generate_Clark_and_Wright, allow_reversal=True),
    "nearest_neighbor": generate_nearest_neighbor,
    "cheapest_insertion": cheapest_insertion,
    "path_cheapest_arc": path_cheapest_arc,
    "farthest_insertion": farthest_insertion,
}


# ============================================================================
# ABSTRACT BASE CLASS
# ============================================================================


class Problem(ABC):
    """
    Abstract Interface for Optimization Problems.
    Defines the contract for state generation, cost calculation, and updates.
    """

    def __init__(self, device: str = "cpu") -> None:
        self.device = device
        self.generator = torch.Generator(device=device)

    def manual_seed(self, seed: int) -> None:
        """Sets the random seed for reproducibility."""
        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(seed)

    @abstractmethod
    def cost(self, solution: torch.Tensor) -> torch.Tensor:
        """Calculate the scalar cost of a solution."""
        pass

    @abstractmethod
    def update(
        self, solution: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply an action to the solution and return (new_solution, is_valid)."""
        pass

    @abstractmethod
    def set_params(self, params: dict) -> None:
        """Set specific instance parameters (coords, demands, etc.)."""
        pass

    @abstractmethod
    def generate_params(
        self, coords: torch.Tensor, demands: torch.Tensor, capacities: torch.Tensor
    ) -> None:
        """Generate a new batch of problem instances."""
        pass

    @property
    def state_encoding(self) -> torch.Tensor:
        """Returns the static encoding of the problem (e.g., coordinates)."""
        return torch.Tensor()

    @abstractmethod
    def generate_init_state(self) -> torch.Tensor:
        """Generates the initial solution/state."""
        pass

    def to_state(self, *components: torch.Tensor) -> torch.Tensor:
        """Concatenates feature tensors into a single state tensor."""
        return torch.cat(components, dim=-1)

    def from_state(self, state: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Splits the state tensor back into its components."""
        return state[:, :, 0].unsqueeze(-1), state[:, :, 1:]


# ============================================================================
# CVRP IMPLEMENTATION
# ============================================================================


class CVRP(Problem):
    """
    Capacitated Vehicle Routing Problem (CVRP).
    Optimization goal: Minimize total route distance subject to vehicle capacity constraints.
    """

    x_dim = 1  # Dimension of the solution (node index)

    def __init__(
        self,
        dim: int = 50,
        n_problems: int = 256,
        device: str = "cpu",
        params: dict | None = None,
    ):
        super().__init__(device)
        self.params = params or {}
        self.n_problems = n_problems
        self.dim = dim
        self.heuristic = None
        self.feature_flags = {}

    # ------------------------------------------------------------------------
    # Configuration & Parameter Setup
    # ------------------------------------------------------------------------

    def set_heuristic(self, heuristic_name: str) -> None:
        """Selects the local search heuristic (swap, two_opt, insertion)."""
        heuristics_map = {
            "swap": swap,
            "two_opt": self.apply_two_opt_split,
            "insertion": insertion,
        }
        self.heuristic = heuristics_map.get(heuristic_name)
        if self.heuristic is None:
            raise ValueError(f"Unsupported heuristic: {heuristic_name}")

    def apply_heuristic(self, solution: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Applies the configured heuristic to the solution."""
        if self.heuristic is None:
            raise ValueError("Heuristic not configured. Call set_heuristic() first.")
        return self.heuristic(solution, action)

    def set_params(self, params: dict) -> None:
        """
        Loads batch data (coords, demands, capacity) and pre-computes static features
        like distance matrices, angles, and isolation scores.
        """
        if "coords" in params:
            self.coords = params["coords"].to(self.device)
        if "demands" in params:
            self.demands = params["demands"].to(self.device)
        if "capacity" in params:
            self.capacity = params["capacity"].to(self.device)
        with torch.no_grad():
            # Pre-compute static geometric features
            self.angles = calculate_client_angles(self.coords)
            self.matrix = calculate_distance_matrix(self.coords)
            # Mask ghost/padding nodes (demand==0, index>0) from the distance matrix
            # so they can never appear as KNN neighbours. Real depot (index 0) is kept.
            real_counts = (self.demands > 0).sum(dim=1) + 1  # customers + depot [B]
            min_real = int(real_counts.min().item())
            is_ghost = self.demands == 0
            is_ghost[:, 0] = False  # keep real depot
            masked_matrix = self.matrix.masked_fill(is_ghost.unsqueeze(1), float("inf"))

            self.mean_dist_5 = calculate_knn_isolation(masked_matrix, k=5)
            self.mean_dist_10 = calculate_knn_isolation(masked_matrix, k=max(1, min_real // 10))
            self.mean_dist_33 = calculate_knn_isolation(masked_matrix, k=max(1, min_real // 3))

            # Distance-rank matrix: rank_matrix[b, i, j] is the normalized rank of node j
            # among node i's neighbors sorted by distance (self=0, closest real node=1/N).
            # Ghosts (inf in masked_matrix) sort last; N excludes self.
            ranks = masked_matrix.argsort(dim=-1).argsort(dim=-1)  # [B, N+1, N+1]
            denom = (real_counts - 1).clamp(min=1).view(-1, 1, 1)
            self.rank_matrix = (ranks.float() / denom).clamp(max=1.0)

            # Normalize distances from depot [0, 1]
            self.dist_to_depot = self.matrix[:, 0, 0:]
            min_dist = torch.min(self.dist_to_depot, dim=1, keepdim=True)[0]
            max_dist = torch.max(self.dist_to_depot, dim=1, keepdim=True)[0]
            divisor = torch.clamp(max_dist - min_dist, min=1e-10)
            self.dist_to_depot = ((self.dist_to_depot - min_dist) / divisor).unsqueeze(-1)

            self.depot_coords = self.coords[:, 0, :].unsqueeze(1)
            self.demand_normalized = (self.demands / self.capacity).unsqueeze(-1)

            # Demand normalized by the largest demand in the instance (depot/ghosts -> 0)
            max_demand = self.demands.max(dim=1, keepdim=True)[0].clamp(min=1e-10)
            self.demand_max_normalized = (self.demands / max_demand).unsqueeze(-1)

    def generate_params(
        self, coords: torch.Tensor, demands: torch.Tensor, capacities: torch.Tensor
    ) -> None:
        """Validates input shapes and sets problem parameters."""
        for name, tensor in [
            ("coords", coords),
            ("demands", demands),
            ("capacities", capacities),
        ]:
            if tensor.shape[0] != self.n_problems:
                raise ValueError(f"Expected {self.n_problems} for {name}, got {tensor.shape[0]}")

        self.set_params({"coords": coords, "demands": demands, "capacity": capacities})

    # ------------------------------------------------------------------------
    # Feature Engineering & State Construction
    # ------------------------------------------------------------------------

    def set_feature_flags(self, feature_flags: dict[str, bool]) -> None:
        self.feature_flags = feature_flags

    def get_input_dim(self) -> int:
        """Calculates input channel dimension based on active feature flags."""
        dims = {
            "static": 6,  # x, y, th, d, is_depot, q/Q
            "topology": 4,  # prev_x, prev_y, next_x, next_y
            "neighbor_rank": 2,  # distance-rank of prev/next neighbors
            "demand_max": 1,  # demand / largest demand in instance
            "density5": 1,  # mean_dist_5
            "density10%": 1,  # mean_dist_10%
            "density33%": 1,  # mean_dist_33%
            "detour": 1,
            "centroid": 1,
            # "route_cost": 1,
            "route_pct": 1,
            "slack": 1,
            "node_pct": 1,
            "meta": 2,  # temp, progress
        }
        return sum(dims[k] for k, v in self.feature_flags.items() if v)

    def get_distance_to_centroid(self, solution: torch.Tensor) -> torch.Tensor:
        """Computes distance of each node to its route's center of gravity."""
        coords = self.get_coords(solution)

        # Setup aggregation tensors. seq_len + 1 is a static upper bound on the number
        # of routes (segment_ids = mask.cumsum, max <= seq_len); extra columns stay zero
        # and are never gathered. Avoids the per-step .item() host<->device sync.
        batch_size, seq_len = self.segment_ids.shape
        num_routes = seq_len + 1

        route_coord_sums = torch.zeros(
            batch_size, num_routes, 2, device=self.device, dtype=coords.dtype
        )
        route_node_counts = torch.zeros(
            batch_size, num_routes, 1, device=self.device, dtype=coords.dtype
        )

        # Expand segment IDs for gathering
        segment_ids_expanded = self.segment_ids.unsqueeze(-1).expand(-1, -1, 2)

        # Aggregate coordinates and counts per route
        route_coord_sums.scatter_add_(1, segment_ids_expanded, coords)

        ones = torch.ones(batch_size, seq_len, 1, device=self.device, dtype=coords.dtype)
        # Count only valid nodes (using segment_ids to group)
        route_node_counts.scatter_add_(1, self.segment_ids.unsqueeze(-1), ones)

        # Calculate Centroids
        route_node_counts = torch.clamp(route_node_counts, min=1.0)
        route_centroids = route_coord_sums / route_node_counts

        # Map centroids back to node positions
        node_centroids = route_centroids.gather(1, segment_ids_expanded)

        return (coords - node_centroids).norm(p=2, dim=-1, keepdim=True)

    def build_state_components(
        self, x: torch.Tensor, temp: torch.Tensor, time: torch.Tensor
    ) -> list[torch.Tensor]:
        """Assembles the state vector from static, topological, and dynamic features."""
        flags = self.feature_flags
        components = [x]

        # Prepare coordinates with padding for gathering
        padding = max(0, x.size(1) - self.state_encoding.size(1))
        padded_coords = F.pad(self.state_encoding, (0, 0, 0, padding)).gather(
            1, x.expand(-1, -1, 2)
        )

        # 1. Static Features
        if flags.get("static", True):
            components.extend(
                [
                    padded_coords,
                    (x == 0).long(),  # is_depot
                    self.angles.gather(1, x),  # theta
                    self.dist_to_depot.gather(1, x),  # dist to depot
                    self.demand_normalized.gather(1, x),
                ]
            )

        # 2. Topology (Immediate Neighbors)
        if flags.get("topology", True):
            components.append(torch.roll(padded_coords, shifts=1, dims=1))  # Prev
            components.append(torch.roll(padded_coords, shifts=-1, dims=1))  # Next

        # 2b. Neighbor distance-rank (rank of prev/next neighbor from current node)
        if flags.get("neighbor_rank", False):
            components.append(calculate_neighbor_rank_features(x, self.rank_matrix))

        # 2c. Demand normalized by largest demand in the instance
        if flags.get("demand_max", False):
            components.append(self.demand_max_normalized.gather(1, x))

        # 3. Density Features
        if flags.get("density5", False):
            components.append(self.mean_dist_5.gather(1, x))
        if flags.get("density10%", False):  # Defaults to True if you want them always on
            components.append(self.mean_dist_10.gather(1, x))
        if flags.get("density33%", False):
            components.append(self.mean_dist_33.gather(1, x))

        # 4. Local Cost Features
        if flags.get("detour", False):
            components.append(calculate_detour_features(x, self.matrix))
        if flags.get("centroid", False):
            components.append(self.get_distance_to_centroid(x))

        # 5. Route Status (Capacity & Load)
        if any(flags.get(k) for k in ["route_pct", "slack", "node_pct"]):
            node_pct, route_pct, slack = self.get_percentage_demands()
            if flags.get("route_pct", False):
                components.append(route_pct)
            if flags.get("slack", False):
                components.append(slack)
            if flags.get("node_pct", False):
                components.append(node_pct)

        # 6. Route Cost Normalized
        # if flags.get("route_cost", False):
        #     components.append(self.cost_per_route(x))

        # 7. Metadata
        if flags.get("meta", True):
            components.extend([repeat_to(temp, x), repeat_to(time, x)])

        return components

    def conditional_neighbor_rank(self, x: torch.Tensor, c1: torch.Tensor) -> torch.Tensor:
        """
        Distance-rank features describing the two edges an *insertion* move creates.

        The `insertion` operator moves the chosen first node u (city 1) to sit
        immediately after a candidate position v (city 2), so it creates edges
        (v, u) and (u, succ(v)). This returns the bidirectional rank of both new
        edges, giving the policy information about the full local neighbourhood of
        the insertion -- not just the (u, v) pair.

        NOTE: these semantics match the `insertion` heuristic only. For `swap`/
        `two_opt` the successor edge does not correspond to the move's geometry.

        Args:
            x: [B, L, 1] (or [B, L]) node-index sequence (state column 0 / solution).
            c1: [B] chosen first-node *position* in the sequence (not a node id).

        Returns:
            [B, L, 4] = (rank[u, v], rank[v, u], rank[u, succ(v)], rank[succ(v), u])
            for every candidate position, with u = node at position c1, v = node at
            each position, and succ(v) = node at the next position (wrapping at the
            end). Values in [0, 1]. Assumes c1 indexes the same sequence as x (true
            for the 'valid'/'free' update methods; not 'rm_depot', which compacts
            positions).
        """
        nodes = x.squeeze(-1).long()  # [B, L] node id at each position
        arange = torch.arange(nodes.size(0), device=nodes.device)
        u = nodes[arange, c1]  # [B] node id at the chosen first position
        succ = torch.roll(nodes, shifts=-1, dims=1)  # [B, L] node after each position

        rank_uv = self.rank_matrix[arange, u].gather(1, nodes)  # rank of v from u's view
        rank_vu = self.rank_matrix[arange, :, u].gather(1, nodes)  # rank of u from v's view
        rank_us = self.rank_matrix[arange, u].gather(1, succ)  # rank of succ(v) from u's view
        rank_su = self.rank_matrix[arange, :, u].gather(1, succ)  # rank of u from succ(v)'s view
        return torch.stack([rank_uv, rank_vu, rank_us, rank_su], dim=-1)  # [B, L, 4]

    def conditional_insertion_detour(self, x: torch.Tensor, c1: torch.Tensor) -> torch.Tensor:
        """
        Cost of inserting the chosen first node u (city 1) after each candidate position.

        The `insertion` operator moves u to sit between a candidate position v (city 2)
        and its successor succ(v). This returns, for every candidate position:
          - insert_cost = d(v, u) + d(u, succ(v)) - d(v, succ(v))   (>= 0)
                The detour u creates at its new location (cheapest-insertion delta).
          - net_delta   = insert_cost - removal_gain(u)
                The actual change in total tour length of the full move, where
                removal_gain(u) = d(prev(u), u) + d(u, succ(u)) - d(prev(u), succ(u))
                is the saving from pulling u out of its current spot (constant over v).
                net_delta < 0 means the move shortens the tour.

        NOTE: insertion-specific, like conditional_neighbor_rank. Raw (un-normalized)
        distances, matching calculate_detour_features.

        Args:
            x: [B, L, 1] (or [B, L]) node-index sequence (state column 0 / solution).
            c1: [B] chosen first-node *position* in the sequence (not a node id).

        Returns:
            [B, L, 2] = (insert_cost, net_delta) for every candidate position.
        """
        nodes = x.squeeze(-1).long()  # [B, L] node id at each position
        arange = torch.arange(nodes.size(0), device=nodes.device)
        u = nodes[arange, c1]  # [B] node id at the chosen first position

        prev = torch.roll(nodes, shifts=1, dims=1)  # [B, L] node before each position
        succ = torch.roll(nodes, shifts=-1, dims=1)  # [B, L] node after each position

        # Insertion cost at every candidate position v: d(v,u) + d(u,succ(v)) - d(v,succ(v))
        b = arange[:, None]  # [B, 1] broadcasts over the L positions
        d_vu = self.matrix[b, nodes, u[:, None]]  # d(v, u)
        d_us = self.matrix[b, u[:, None], succ]  # d(u, succ(v))
        d_vs = self.matrix[b, nodes, succ]  # d(v, succ(v))
        insert_cost = d_vu + d_us - d_vs  # [B, L]

        # Removal gain of u at its current position (scalar per instance, broadcast over L)
        prev_u = prev[arange, c1]  # node before u
        succ_u = succ[arange, c1]  # node after u
        removal_gain = (
            self.matrix[arange, prev_u, u]
            + self.matrix[arange, u, succ_u]
            - self.matrix[arange, prev_u, succ_u]
        )  # [B]
        net_delta = insert_cost - removal_gain[:, None]  # [B, L]

        return torch.stack([insert_cost, net_delta], dim=-1)  # [B, L, 2]

    # ------------------------------------------------------------------------
    # Initialization & Helpers
    # ------------------------------------------------------------------------

    @property
    def state_encoding(self) -> torch.Tensor:
        return self.coords

    def get_coords(self, solution: torch.Tensor) -> torch.Tensor:
        """Retrieves coordinates for nodes in the solution sequence."""
        return torch.gather(self.coords, 1, solution.expand(-1, -1, self.coords.size(-1)))

    def get_demands(self, solution: torch.Tensor) -> torch.Tensor:
        """Retrieves demands for nodes in the solution sequence."""
        return torch.gather(self.demands, 1, solution.squeeze(-1))

    def generate_init_state(
        self,
        init_heuristic: str = "",
        multi_init: bool = False,
        init_list: list[str] = [],
    ) -> torch.Tensor:
        """Generates the initial population of solutions."""
        if multi_init:
            # Generate sub-batches with different heuristics and concatenate
            split_size = self.n_problems // len(init_list)
            solutions = []

            for i, method in enumerate(init_list):
                raw_sol = INIT_METHODS[method](self).to(self.device)
                start_idx = i * split_size
                end_idx = (i + 1) * split_size if i < len(init_list) - 1 else self.n_problems
                solutions.append(raw_sol[start_idx:end_idx])

            # Pad to match largest solution
            max_size = max(s.shape[1] for s in solutions)
            solutions_padded = [F.pad(s, (0, 0, 0, max_size - s.shape[1])) for s in solutions]
            sol = torch.cat(solutions_padded, dim=0)
        else:
            if init_heuristic not in INIT_METHODS:
                raise ValueError(f"Unsupported init method: {init_heuristic}")
            sol = INIT_METHODS[init_heuristic](self).to(self.device)

        return self.init_parameters(sol)

    def init_parameters(self, solution: torch.Tensor) -> torch.Tensor:
        """Initializes internal bookkeeping tensors (segment_ids, mask) for the given solution."""
        self.ordered_demands = self.get_demands(solution)

        if not is_feasible(solution, self.ordered_demands, self.capacity).all():
            raise ValueError("Generated initial solution is not feasible.")

        # Identify route segments based on zero-demand delimiters (depots)

        self.mask = self.ordered_demands == 0
        self.segment_ids = self.mask.long().cumsum(dim=1)

        return solution

    def update_tensor(self, solution: torch.Tensor) -> None:
        """Updates internal bookkeeping tensors when the solution changes."""
        self.ordered_demands = self.get_demands(solution)
        # self.mask = self.ordered_demands != 0
        # segment_start = self.mask & ~torch.cat(
        #     [torch.zeros_like(self.mask[:, :1]), self.mask[:, :-1]], dim=1
        # )
        # self.segment_ids = torch.cumsum(segment_start, 1) * self.mask
        self.mask = self.ordered_demands == 0
        self.segment_ids = self.mask.long().cumsum(dim=1)

    # ------------------------------------------------------------------------
    # Cost & Load Calculations
    # ------------------------------------------------------------------------

    def cost(self, solution: torch.Tensor) -> torch.Tensor:
        """Computes total tour length (Euclidean)."""
        return torch.sum(self.get_edge_lengths_in_tour(solution), -1)

    def cost_per_route(self, solution: torch.Tensor) -> torch.Tensor:
        """Computes normalized cost per specific route segment."""
        edge_lengths = self.get_edge_lengths_in_tour(solution)
        total_cost = torch.sum(edge_lengths, -1, keepdim=True)

        # Sum lengths per segment ID
        segment_sums = torch.zeros_like(edge_lengths)
        segment_sums.scatter_add_(1, self.segment_ids, edge_lengths)

        # Broadcast back to nodes
        route_costs = segment_sums.gather(1, self.segment_ids) * ~self.mask
        num_routes = self.segment_ids.max(dim=1, keepdim=True)[0]

        return (route_costs * (num_routes / total_cost)).unsqueeze(-1)

    def get_edge_lengths_in_tour(self, solution: torch.Tensor) -> torch.Tensor:
        """Computes Euclidean distance between node[i] and node[i+1]."""
        coords = self.get_coords(solution)
        next_coords = torch.cat([coords[:, 1:, :], coords[:, :1, :]], dim=1)
        return (coords - next_coords).norm(p=2, dim=-1)

    def get_percentage_demands(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Calculates demand ratios for state representation."""
        node_demands = self.ordered_demands

        # Sum demands per route
        total_demand_per_route = torch.zeros_like(node_demands)
        total_demand_per_route.scatter_add_(1, self.segment_ids, node_demands)
        route_load = total_demand_per_route.gather(1, self.segment_ids)

        # 1. Node demand / Route Load
        node_frac_route = torch.nan_to_num(node_demands / route_load, nan=0.0)

        # 2. Route Load / Vehicle Capacity
        load_frac_capacity = route_load / self.capacity

        # 3. Remaining Capacity / Vehicle Capacity
        remaining_frac_capacity = (self.capacity - route_load) / self.capacity

        return (
            node_frac_route.unsqueeze(-1),
            load_frac_capacity.unsqueeze(-1),
            remaining_frac_capacity.unsqueeze(-1),
        )

    def _get_current_route_loads(self) -> torch.Tensor:
        """Helper to get total load per route ID."""
        # Over-allocate to seq_len + 1 (a static upper bound on num_routes, since
        # segment_ids = mask.cumsum has max <= seq_len). Extra columns stay zero and
        # are never gathered, so the result is identical to sizing by max()+1 -- but we
        # avoid the .item() host<->device sync on the GPU-hot path (called every step).
        num_routes = self.segment_ids.shape[1] + 1
        route_loads = torch.zeros(
            self.n_problems,
            num_routes,
            device=self.device,
            dtype=self.ordered_demands.dtype,
        )
        route_loads.scatter_add_(1, self.segment_ids, self.ordered_demands)
        return route_loads

    # ------------------------------------------------------------------------
    # Solution Update & Logic
    # ------------------------------------------------------------------------

    def update(
        self, solution: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Modifies the solution using the selected heuristic.
        Handles 'rm_depot' experimental mode vs standard mode.
        Checks feasibility and reverts if invalid.
        """
        # Experimental: Remove depot, apply TSP move, re-insert depots
        if self.params.get("UPDATE_METHOD") == "rm_depot":
            mask = solution.squeeze(-1) != 0
            compact_sol = solution[mask].view(solution.size(0), -1, solution.size(-1))

            modified_compact = self.apply_heuristic(compact_sol, action).long()
            sol = construct_cvrp_solution(modified_compact, self.demands, self.capacity)

            # Pad back to original size
            padding_size = solution.size(1) - sol.size(1)
            if padding_size > 0:
                padding = torch.zeros(
                    sol.size(0),
                    padding_size,
                    sol.size(2),
                    dtype=sol.dtype,
                    device=sol.device,
                )
                sol = torch.cat([sol, padding], dim=1)

        # Standard: Direct modification
        else:
            sol = self.apply_heuristic(solution, action).long()

        # Feasibility Check
        valid = torch.ones(sol.size(0), device=self.device, dtype=torch.bool).unsqueeze(-1)
        if self.params.get("UPDATE_METHOD") == "free":
            new_demands = self.get_demands(sol)
            valid = is_feasible(sol, new_demands, self.capacity).unsqueeze(-1).long()

            # Revert invalid moves
        final_sol = torch.where(valid.unsqueeze(-1) == 1, sol, solution).to(torch.int64)

        return final_sol, valid

    def _get_prefix_loads(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Computes load accumulated from the start of the route up to each node (head_load)
        and load remaining from the node to the end of the route (tail_load).

        O(L) segmented cumulative sum. `segment_ids` is monotonic non-decreasing and
        demands are >= 0, so the global exclusive prefix sum is non-decreasing and its
        minimum within a segment is reached at that segment's first node. Subtracting
        that per-segment offset from the (inclusive) global cumsum yields the in-segment
        head load. This replaces the previous O(L^2) [B, L, L] mask matrix (bit-identical
        output, but avoids the L x L memory blow-up that is prohibitive for X/XL sizes).
        """
        d = self.ordered_demands  # [B, L]
        gc = torch.cumsum(d, dim=1)  # inclusive global prefix sum
        seg = self.segment_ids  # [B, L], monotonic non-decreasing
        # seq_len + 1 upper-bounds num routes (seg = mask.cumsum, max <= seq_len). Unused
        # columns keep the sentinel / zero and are never gathered -> identical output,
        # no .item() host<->device sync.
        num_seg = seg.shape[1] + 1

        # Exclusive global prefix sum (cumsum shifted right by one).
        gc_prev = torch.cat([torch.zeros_like(gc[:, :1]), gc[:, :-1]], dim=1)

        # Per-segment offset = exclusive prefix sum at the segment's first node.
        # Sentinel must exceed any real prefix sum; demands may be integer or float.
        sentinel = float("inf") if d.is_floating_point() else torch.iinfo(d.dtype).max
        seg_start = torch.full((d.size(0), num_seg), sentinel, device=d.device, dtype=d.dtype)
        seg_start.scatter_reduce_(1, seg, gc_prev, reduce="amin", include_self=True)
        head_load = gc - seg_start.gather(1, seg)  # in-segment cumulative load up to i (incl.)

        # Tail = total segment load - head (load strictly after i within the route).
        totals = torch.zeros_like(seg_start).scatter_add_(1, seg, d)
        tail_load = totals.gather(1, seg) - head_load

        return head_load, tail_load

    def get_action_mask(self, solution: torch.Tensor, node_pos: torch.Tensor) -> torch.Tensor:
        """
        Determines valid insertion moves based on the current state.
        State tensors (segment_ids, ordered_demands) must be up-to-date via update_tensor().
        """
        batch_size, seq_len = self.segment_ids.shape
        node_pos_expanded = node_pos.unsqueeze(-1)  # [batch, 1]

        # --- 1. Common Setup & Gathering ---
        # Identify Routes
        target_route_ids = self.segment_ids
        source_route_id = torch.gather(target_route_ids, 1, node_pos_expanded)

        # Identify Source Validity
        source_demand = torch.gather(self.ordered_demands, 1, node_pos_expanded)
        is_source_valid = source_demand > 0

        # Base Condition: Intra-Route moves are generally valid across heuristics
        # (Reordering within the same route doesn't change total route load)
        is_intra = target_route_ids == source_route_id

        # Initialize mask
        mask = torch.zeros(batch_size, seq_len, device=self.device, dtype=torch.bool)

        # --- 2. Heuristic Specific Logic ---

        if self.heuristic == insertion:
            # Gather Load Info
            per_route_loads = self._get_current_route_loads()
            target_route_loads = torch.gather(per_route_loads, 1, target_route_ids)

            # Capacity Check: Load + New Node <= Q
            potential_loads = target_route_loads + source_demand
            is_capacity_valid = potential_loads <= self.capacity

            # Combine: (Same Route) OR (Fits Capacity)
            mask = is_intra | is_capacity_valid

            # Constraint: Cannot insert after last token (padding/terminator)
            mask[:, -1] = False

        elif self.heuristic == swap:
            # Gather Load Info
            per_route_loads = self._get_current_route_loads()
            target_route_loads = torch.gather(per_route_loads, 1, target_route_ids)
            source_route_load = torch.gather(per_route_loads, 1, source_route_id)

            # Capacity Check: Both routes must fit after swap
            # New Source: Old_Source - Src_Node + Tgt_Node
            new_source_load = source_route_load - source_demand + self.ordered_demands
            # New Target: Old_Target - Tgt_Node + Src_Node
            new_target_load = target_route_loads - self.ordered_demands + source_demand

            is_capacity_valid = (new_source_load <= self.capacity) & (
                new_target_load <= self.capacity
            )

            # A. Standard Swap: (Same Route OR Fits Capacity) AND (Target is Customer)
            standard_swap_mask = (is_intra | is_capacity_valid) & (self.ordered_demands > 0)

            # B. New Route Creation (Triad Pattern [0, 0, 0])
            # Check neighbors for [Depot, Depot, Depot] pattern
            is_depot = self.ordered_demands == 0
            is_prev_depot = torch.roll(is_depot, shifts=1, dims=1)
            is_prev_depot[:, 0] = False  # Fix roll wrap
            is_next_depot = torch.roll(is_depot, shifts=-1, dims=1)
            is_next_depot[:, -1] = False  # Fix roll wrap

            new_route_mask = is_prev_depot & is_depot & is_next_depot

            # Combine
            mask = standard_swap_mask | new_route_mask

        elif self.heuristic == self.apply_two_opt_split:
            # Gather Load Info (Prefix/Suffix)
            head_loads, tail_loads = self._get_prefix_loads()
            source_head = torch.gather(head_loads, 1, node_pos_expanded)
            source_tail = torch.gather(tail_loads, 1, node_pos_expanded)

            # Intra-Route: Standard 2-Opt Reversal
            # Constraint: Both Source and Target must be CUSTOMERS.
            # If we include a Depot in the reversal, we break the Depot->C->C structure.
            is_target_customer = self.ordered_demands > 0
            is_same_route_customer = is_intra & is_target_customer

            # Capacity Check (Tail Swap Logic)
            # Route A: Source Head + Target Tail
            # Route B: Target Head + Source Tail
            new_load_a = source_head + tail_loads
            new_load_b = head_loads + source_tail

            is_capacity_valid = (new_load_a <= self.capacity) & (new_load_b <= self.capacity)

            # Target can be Customer OR Depot (grafting onto a depot is valid)
            is_inter = (~is_intra) & is_capacity_valid

            mask = is_same_route_customer | is_inter

            # Constraint: Remove immediate neighbors (Trivial moves)
            prev_idx = (node_pos_expanded - 1).clamp(min=0)
            next_idx = (node_pos_expanded + 1).clamp(max=seq_len - 1)
            mask.scatter_(1, prev_idx, False)
            mask.scatter_(1, next_idx, False)

            # Constraint: Cannot interact with the padding at the end of sequence
            mask[:, -1] = False

        else:
            raise NotImplementedError(f"Unknown heuristic: {self.heuristic}")

        # --- 3. Global Safety Constraints ---

        # Rule: Cannot select itself (No-Op handled below, not here)
        mask.scatter_(1, node_pos_expanded, False)

        # Force No-Op if trapped
        # (Source is invalid e.g. depot, OR no valid moves in mask)
        has_valid_moves = mask.any(dim=1, keepdim=True)
        force_no_op = (~is_source_valid) | (~has_valid_moves)

        # Construct No-Op mask (Only allow selecting self)
        no_op_mask = torch.zeros_like(mask)
        no_op_mask.scatter_(1, node_pos_expanded, True)

        return torch.where(force_no_op, no_op_mask, mask)

    def apply_two_opt_split(self, solution: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Dispatches to standard 2-opt (intra) or 2-opt-star (inter).
        """
        u, v = action[:, 0], action[:, 1]

        u_seg = self.segment_ids.gather(1, u.unsqueeze(1)).squeeze(1)
        v_seg = self.segment_ids.gather(1, v.unsqueeze(1)).squeeze(1)

        is_intra = u_seg == v_seg

        # We must process the batch. Since we can't easily branch per-sample
        # in a single tensor op without masking, we calculate both and blend.
        # (Or execute purely based on indices if one function handled both,
        # but the logic differs).

        if self.params.get("UPDATE_METHOD") == "rm_depot":
            # In this mode, we have already removed depots and are working on compacted TSP sequences.
            # The action indices correspond to the compacted sequences, so we can directly apply 2-opt logic.
            return two_opt(solution, action)

        # 1. Apply Intra-Route (Standard Reversal)
        sol_intra = two_opt(solution, action)

        # 2. Apply Inter-Route (Tail Swap)
        sol_inter = two_opt_star(solution, action, self.segment_ids)

        # 3. Select based on mask
        # Expand mask to [B, N, 1] to broadcast over solution nodes
        mask = is_intra.view(-1, 1, 1)
        return torch.where(mask, sol_intra, sol_inter)
