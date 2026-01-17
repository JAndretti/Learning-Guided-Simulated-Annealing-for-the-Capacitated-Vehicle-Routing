# ============================================================================
# PROBLEM DEFINITION: CAPACITATED VEHICLE ROUTING PROBLEM (CVRP)
# ============================================================================

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

import torch
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
)
from utils import (
    calculate_client_angles,
    calculate_detour_features,
    calculate_distance_matrix,
    calculate_knn_isolation,
    is_feasible,
    repeat_to,
)

# Registry of initialization methods
INIT_METHODS = {
    "random": random_init_batch,
    "sweep": generate_sweep_solution,
    "isolate": generate_isolate_solution,
    "Clark_and_Wright": generate_Clark_and_Wright,
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
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply an action to the solution and return (new_solution, is_valid)."""
        pass

    @abstractmethod
    def set_params(self, params: Dict) -> None:
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

    def from_state(self, state: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """Splits the state tensor back into its components."""
        num_extra_features = state.shape[-1] - 3  # Adjust based on dynamic features
        split_sizes = [1, 2] + [1] * num_extra_features
        return tuple(torch.split(state, split_sizes, dim=-1))


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
        params: Optional[Dict] = None,
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
            "two_opt": two_opt,
            "insertion": insertion,
        }
        self.heuristic = heuristics_map.get(heuristic_name)
        if self.heuristic is None:
            raise ValueError(f"Unsupported heuristic: {heuristic_name}")

    def apply_heuristic(
        self, solution: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        """Applies the configured heuristic to the solution."""
        if self.heuristic is None:
            raise ValueError("Heuristic not configured. Call set_heuristic() first.")
        return self.heuristic(solution, action)

    def set_params(self, params: Dict) -> None:
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
            self.isolation_score = calculate_knn_isolation(self.matrix, k=5)

            # Normalize distances from depot [0, 1]
            self.dist_to_depot = self.matrix[:, 0, 0:]
            min_dist = torch.min(self.dist_to_depot, dim=1, keepdim=True)[0]
            max_dist = torch.max(self.dist_to_depot, dim=1, keepdim=True)[0]
            divisor = torch.clamp(max_dist - min_dist, min=1e-10)
            self.dist_to_depot = ((self.dist_to_depot - min_dist) / divisor).unsqueeze(
                -1
            )

            self.depot_coords = self.coords[:, 0, :].unsqueeze(1)
            self.demand_normalized = (self.demands / self.capacity).unsqueeze(-1)
            # self.ref_cost = torch.mean(self.cost(INIT_METHODS["nearest_neighbor"](self))).to(
            #     self.device
            # )

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
                raise ValueError(
                    f"Expected {self.n_problems} for {name}, got {tensor.shape[0]}"
                )

        self.set_params({"coords": coords, "demands": demands, "capacity": capacities})

    # ------------------------------------------------------------------------
    # Feature Engineering & State Construction
    # ------------------------------------------------------------------------

    def set_feature_flags(self, feature_flags: Dict[str, bool]) -> None:
        self.feature_flags = feature_flags

    def get_input_dim(self) -> int:
        """Calculates input channel dimension based on active feature flags."""
        dims = {
            "static": 7,  # x, y, th, d, is_depot, knn, q/Q
            "topology": 4,  # prev_x, prev_y, next_x, next_y
            # "gap_ref": 1,
            "detour": 1,
            "centroid": 1,
            "route_status": 4,  # load/Q, slack, q/load, route_cost_norm
            "route_cost": 1,
            "route_pct": 1,
            "slack": 1,
            "node_pct": 1,
            "meta": 2,  # temp, progress
        }
        return sum(dims[k] for k, v in self.feature_flags.items() if v)

    def get_distance_to_centroid(self, solution: torch.Tensor) -> torch.Tensor:
        """Computes distance of each node to its route's center of gravity."""
        coords = self.get_coords(solution)

        # Setup aggregation tensors
        num_routes = int(self.segment_ids.max().item()) + 1
        batch_size, seq_len = self.segment_ids.shape

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

        ones = torch.ones(
            batch_size, seq_len, 1, device=self.device, dtype=coords.dtype
        )
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
    ) -> List[torch.Tensor]:
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
                    self.isolation_score.gather(1, x),
                ]
            )

        # 2. Topology (Immediate Neighbors)
        if flags.get("topology", True):
            components.append(torch.roll(padded_coords, shifts=1, dims=1))  # Prev
            components.append(torch.roll(padded_coords, shifts=-1, dims=1))  # Next

        # if flags.get("gap_ref", False):
        #     current_cost = self.cost(x).unsqueeze(-1)

        #     # Normalized Gap
        #     # > 0 : Worse than baseline
        #     # 0   : Equal to baseline
        #     # < 0 : Better than baseline
        #     gap = (current_cost - self.ref_cost) / self.ref_cost
        #     components.append(gap)

        # 3. Local Cost Features
        if flags.get("detour", True):
            components.append(calculate_detour_features(x, self.matrix))
        if flags.get("centroid", False):
            components.append(self.get_distance_to_centroid(x))

        # 4. Route Status (Capacity & Load)
        if any(flags.get(k) for k in ["route_pct", "slack", "node_pct"]):
            node_pct, route_pct, slack = self.get_percentage_demands()
            if flags.get("route_pct", True):
                components.append(route_pct)
            if flags.get("slack", False):
                components.append(slack)
            if flags.get("node_pct", False):
                components.append(node_pct)

        if flags.get("route_cost", False):
            components.append(self.cost_per_route(x))

        # 5. Metadata
        if flags.get("meta", True):
            components.extend([repeat_to(temp, x), repeat_to(time, x)])

        return components

    # ------------------------------------------------------------------------
    # Initialization & Helpers
    # ------------------------------------------------------------------------

    @property
    def state_encoding(self) -> torch.Tensor:
        return self.coords

    def get_coords(self, solution: torch.Tensor) -> torch.Tensor:
        """Retrieves coordinates for nodes in the solution sequence."""
        return torch.gather(
            self.coords, 1, solution.expand(-1, -1, self.coords.size(-1))
        )

    def get_demands(self, solution: torch.Tensor) -> torch.Tensor:
        """Retrieves demands for nodes in the solution sequence."""
        return torch.gather(self.demands, 1, solution.squeeze(-1))

    def generate_init_state(
        self,
        init_heuristic: str = "",
        multi_init: bool = False,
        init_list: List[str] = [],
    ) -> torch.Tensor:
        """Generates the initial population of solutions."""
        if multi_init:
            # Generate sub-batches with different heuristics and concatenate
            split_size = self.n_problems // len(init_list)
            solutions = []

            for i, method in enumerate(init_list):
                raw_sol = INIT_METHODS[method](self).to(self.device)
                start_idx = i * split_size
                end_idx = (
                    (i + 1) * split_size if i < len(init_list) - 1 else self.n_problems
                )
                solutions.append(raw_sol[start_idx:end_idx])

            # Pad to match largest solution
            max_size = max(s.shape[1] for s in solutions)
            solutions_padded = [
                F.pad(s, (0, 0, 0, max_size - s.shape[1])) for s in solutions
            ]
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
        # self.mask = self.ordered_demands != 0
        # segment_start = self.mask & ~torch.cat(
        #     [torch.zeros_like(self.mask[:, :1]), self.mask[:, :-1]], dim=1
        # )
        # self.segment_ids = torch.cumsum(segment_start, 1) * self.mask

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
        route_costs = segment_sums.gather(1, self.segment_ids) * self.mask
        num_routes = self.segment_ids.max(dim=1, keepdim=True)[0]

        return (route_costs * (num_routes / total_cost)).unsqueeze(-1)

    def get_edge_lengths_in_tour(self, solution: torch.Tensor) -> torch.Tensor:
        """Computes Euclidean distance between node[i] and node[i+1]."""
        coords = self.get_coords(solution)
        next_coords = torch.cat([coords[:, 1:, :], coords[:, :1, :]], dim=1)
        return (coords - next_coords).norm(p=2, dim=-1)

    def get_percentage_demands(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
        num_routes = self.segment_ids.max() + 1
        route_loads = torch.zeros(
            self.n_problems,
            int(num_routes.item()),
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
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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
        new_demands = self.get_demands(sol)
        valid = is_feasible(sol, new_demands, self.capacity).unsqueeze(-1).long()

        if not valid.all():
            print("Warning: Some modified solutions are infeasible.")

        # Revert invalid moves
        final_sol = torch.where(valid.unsqueeze(-1) == 1, sol, solution).to(torch.int64)

        return final_sol, valid

    def get_action_mask(
        self, solution: torch.Tensor, node_pos: torch.Tensor
    ) -> torch.Tensor:
        """
        Determines valid insertion moves based on the current state.
        State tensors (segment_ids, ordered_demands) must be up-to-date via update_tensor().
        """
        batch_size, seq_len = self.segment_ids.shape
        node_pos_expanded = node_pos.unsqueeze(-1)  # [batch, 1]

        # 1. Expand Route Loads to Sequence Length
        # _get_current_route_loads gives [batch, num_routes].
        # We gather this so every position knows the total load of the route it belongs to.
        per_route_loads = self._get_current_route_loads()
        target_route_loads = torch.gather(per_route_loads, 1, self.segment_ids)

        # 2. Get Source Node Information (The node we are moving)
        source_demand = torch.gather(self.ordered_demands, 1, node_pos_expanded)
        source_route_id = torch.gather(self.segment_ids, 1, node_pos_expanded)

        # Check if source is valid (e.g., demand > 0 implies it's a customer, not depot/padding)
        is_source_valid = source_demand > 0

        # 3. Compute Heuristic Mask
        target_route_ids = self.segment_ids
        mask = torch.zeros(batch_size, seq_len, device=self.device, dtype=torch.bool)

        if self.heuristic == insertion:
            # Condition A: Intra-Route Move (Always Valid)
            # Moving a node within its own route does not change the total load.
            is_same_route = target_route_ids == source_route_id

            # Condition B: Inter-Route Move (Capacity Check)
            # If moving to a new route, check if (RouteLoad + NodeDemand) <= Capacity
            potential_loads = target_route_loads + source_demand
            is_capacity_valid = potential_loads <= self.capacity

            # Combine: Valid if (Same Route) OR (Fits Capacity)
            mask = is_same_route | is_capacity_valid

            # Rules:
            # - Cannot insert after itself (no-op)
            mask.scatter_(1, node_pos_expanded, False)
            # - Cannot insert after the last token (terminator/padding)
            mask[:, -1] = False

        elif self.heuristic in [swap, two_opt]:
            raise NotImplementedError(f"{self.heuristic} not implemented in masking.")
        else:
            raise NotImplementedError("Unknown heuristic")

        # 4. Safety / Fallback (Force No-Op if trapped)
        # If the source node is invalid (e.g. depot) OR no valid moves exist,
        # we force the agent to select the node itself (No-Op).

        has_valid_moves = mask.any(dim=1, keepdim=True)
        force_no_op = (~is_source_valid) | (~has_valid_moves)

        # Create a mask that only allows selecting the node itself
        no_op_mask = torch.zeros_like(mask)
        no_op_mask.scatter_(1, node_pos_expanded, True)

        return torch.where(force_no_op, no_op_mask, mask)

    # def get_action_mask(
    #     self, solution: torch.Tensor, node_pos: torch.Tensor
    # ) -> torch.Tensor:
    #     """
    #     Determines valid moves (Insertion Only).

    #     A move is valid if:
    #     1. It is within the same route (Intra-route).
    #     2. It is to a different route AND that route has spare capacity.

    #     Handles special logic for inserting after a depot (implies entering the *next* route).
    #     """
    #     batch_size, num_nodes, _ = solution.shape

    #     # 1. Prepare State Data
    #     current_route_loads = self._get_current_route_loads()
    #     node_pos = node_pos.unsqueeze(-1)

    #     # Source (Node being moved) info
    #     source_demand = torch.gather(self.ordered_demands, 1, node_pos)
    #     source_route_id = torch.gather(self.segment_ids, 1, node_pos)
    #     is_source_valid = source_demand > 0

    #     # Target (Destination position) info
    #     target_route_ids = self.segment_ids
    #     target_route_loads = torch.gather(current_route_loads, 1, target_route_ids)

    #     # 2. Fix Logic for Depots & Route Transitions
    #     # If target is depot (ID=0), we are technically looking at the *next* route.
    #     is_target_depot = target_route_ids == 0

    #     # Look ahead for loads
    #     next_route_loads = torch.roll(target_route_loads, shifts=-1, dims=1)
    #     next_route_loads[:, -1] = 0

    #     effective_target_loads = torch.where(
    #         is_target_depot, next_route_loads, target_route_loads
    #     )

    #     # Look ahead for Route IDs
    #     next_route_ids = torch.roll(target_route_ids, shifts=-1, dims=1)
    #     next_route_ids[:, -1] = 0

    #     effective_target_ids = torch.where(
    #         is_target_depot, next_route_ids, target_route_ids
    #     )

    #     # 3. Compute Heuristic Mask
    #     mask = torch.ones(batch_size, num_nodes, device=self.device, dtype=torch.bool)

    #     if self.heuristic is insertion:
    #         # Condition A: Intra-route move
    #         is_same_route = source_route_id == effective_target_ids

    #         # Condition B: Capacity Check (Destination Load + Source Demand <= Capacity)
    #         new_load = effective_target_loads + source_demand
    #         is_capacity_valid = new_load <= self.capacity

    #         mask = is_same_route | is_capacity_valid

    #         # Rules: Cannot insert after itself, Cannot insert at array end
    #         mask.scatter_(1, node_pos, False)
    #         mask[:, -1] = False

    #     elif self.heuristic in [swap, two_opt]:
    #         raise NotImplementedError(f"{self.heuristic} not implemented in masking.")
    #     else:
    #         raise NotImplementedError("Unknown heuristic")

    #     # 4. Final Safety Constraints
    #     # Allow invalid nodes (e.g. depot) to only perform "No-Op" (select themselves)
    #     no_op_mask = torch.zeros_like(mask)
    #     no_op_mask.scatter_(1, node_pos, True)

    #     has_valid_moves = mask.any(dim=1, keepdim=True)
    #     force_no_op = (~is_source_valid) | (~has_valid_moves)

    #     return torch.where(force_no_op, no_op_mask, mask)
