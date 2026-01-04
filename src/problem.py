# --------------------------------
# Import required libraries
# --------------------------------
from abc import ABC, abstractmethod  # Abstract base classes for problem definition
from typing import (
    Dict,
    Optional,
    Tuple,
    Union,
)  # Type hints for better code readability

import torch  # PyTorch for tensor operations
import torch.nn.functional as F

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

init_methods = {
    "random": random_init_batch,
    "sweep": generate_sweep_solution,
    "isolate": generate_isolate_solution,
    "Clark_and_Wright": generate_Clark_and_Wright,
    "nearest_neighbor": generate_nearest_neighbor,
    "cheapest_insertion": cheapest_insertion,
    "path_cheapest_arc": path_cheapest_arc,
    "farthest_insertion": farthest_insertion,
}


class Problem(ABC):
    """
    Abstract base class defining the interface for optimization problems.

    This class provides the foundation for implementing various optimization problems
    by defining common operations and required abstract methods.
    """

    def __init__(self, device: str = "cpu") -> None:
        """
        Initialize the problem.

        Args:
            device: Computation device (cpu/cuda)
        """
        self.device = device
        self.generator = torch.Generator(device=device)

    def manual_seed(self, seed: int) -> None:
        """
        Set random generator seed for reproducibility.

        Args:
            seed: Random seed value
        """
        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(seed)

    @abstractmethod
    def cost(self, solution: torch.Tensor) -> torch.Tensor:
        """
        Calculate cost of a solution.

        Args:
            solution: Solution tensor

        Returns:
            Cost tensor
        """
        pass

    @abstractmethod
    def update(
        self, solution: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply an action to modify a solution.

        Args:
            s: Current solution tensor
            a: Action tensor

        Returns:
            Updated solution tensor
        """
        pass

    @abstractmethod
    def set_params(self, params) -> None:
        """
        Set problem parameters.

        Args:
            **kwargs: Problem-specific parameters
        """
        pass

    @abstractmethod
    def generate_params(self) -> None:
        """
        Generate problem parameters.

        Returns:
            Dictionary of problem parameters
        """
        pass

    @property
    def state_encoding(self) -> torch.Tensor:
        """
        Get problem's state encoding.

        Returns:
            Tensor representation of the problem state
        """
        return torch.Tensor()

    @abstractmethod
    def generate_init_state(self) -> torch.Tensor:
        """
        Generate initial state for the problem.

        Returns:
            Initial state tensor
        """
        pass

    def to_state(self, *components: torch.Tensor) -> torch.Tensor:
        """
        Concatenate multiple state components into a single state tensor.

        Args:
            *components: Variable number of tensors representing different components
                        of the state. Each tensor should have the same shape except
                        for the last dimension, which will be concatenated.

        Returns:
            A single tensor resulting from concatenating all input components
            along the last dimension.
        """
        return torch.cat(components, dim=-1)

    def from_state(self, state: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        Split state into components dynamically.

        Args:
            state: Combined state tensor

        Returns:
            Tuple of component tensors
        """
        num_extra_features = state.shape[-1] - 3  # Adjusting for variable dimensions
        split_sizes = [1, 2] + [1] * num_extra_features
        return tuple(torch.split(state, split_sizes, dim=-1))


class CVRP(Problem):
    """
    Capacitated Vehicle Routing Problem implementation.

    This class implements the CVRP, where a fleet of vehicles with limited capacity
    must serve customer demands while minimizing total route distance.
    """

    x_dim = 1  # Dimension for solution representation

    def __init__(
        self,
        dim: int = 50,
        n_problems: int = 256,
        device: str = "cpu",
        params: Optional[Dict] = None,
    ):
        """
        Initialize CVRP instance.

        Args:
            dim: Number of client nodes (excluding depot)
            n_problems: Batch size for parallel processing
            capacities: Vehicle capacity constraint(s)
            device: Computation device (cpu/cuda)
            params: Configuration parameters including:
                   - HEURISTIC: 'swap' or 'two_opt', etc.
                   - CLUSTERING: Whether to use clustered instances
                   - NB_CLUSTERS_MAX: Max clusters if clustering enabled
                   - UPDATE_METHOD: How to apply heuristics
                   - INIT: Initial solution generation method
        """
        super().__init__(device)
        self.params = params or {}
        self.n_problems = n_problems
        self.dim = dim

    # --------------------------------
    # Initialization and Configuration
    # --------------------------------

    def set_heuristic(self, heuristic: list) -> None:
        """
        Configure the heuristic method for solution modification.

        Args:
            heuristic: Type of heuristic ('swap', 'two_opt', 'insertion')

        Raises:
            ValueError: If unsupported heuristic specified
        """
        self.heuristic = None
        self.heuristic_1 = None
        self.heuristic_2 = None
        if isinstance(heuristic, (list, str)):
            if isinstance(heuristic, str):
                heuristic = [heuristic]
            heuristics = {
                "swap": swap,
                "two_opt": two_opt,
                "insertion": insertion,
            }
            if len(heuristic) == 1:
                self.heuristic = heuristics.get(heuristic[0])
                if self.heuristic is None:
                    raise ValueError(f"Unsupported heuristic: {heuristic[0]}")
            elif len(heuristic) == 2:
                self.heuristic_1 = heuristics.get(heuristic[0])
                self.heuristic_2 = heuristics.get(heuristic[1])
                if self.heuristic_1 is None or self.heuristic_2 is None:
                    raise ValueError(f"Unsupported heuristics: {heuristic}")
            else:
                raise ValueError("Only up to 2 heuristics are supported.")

    def apply_heuristic(
        self, solution: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply the selected heuristic to the given solution.

        Args:
            solution: Current solution tensor
            action: Action tensor indicating the modification to apply

        Returns:
            Modified solution tensor
        """
        if self.heuristic is not None:
            return self.heuristic(solution, action)
        elif self.heuristic_1 is not None and self.heuristic_2 is not None:
            idx = action[:, 2]
            sol1 = self.heuristic_1(solution, action[:, :2].long())
            sol2 = self.heuristic_2(solution, action[:, :2].long())
            return torch.where(idx.unsqueeze(-1).unsqueeze(-1) == 0, sol1, sol2)
        else:
            raise ValueError("Heuristic not properly configured.")

    def set_params(self, params: Dict) -> None:
        """
        Update problem coordinates and demands.

        Args:
            params: Dictionary containing problem parameters
        """
        if "coords" in params:
            self.coords = params["coords"].to(self.device)
        if "demands" in params:
            self.demands = params["demands"].to(self.device)
        if "capacity" in params:
            self.capacity = params["capacity"].to(self.device)
        self.angles = calculate_client_angles(self.coords)
        self.matrix = calculate_distance_matrix(self.coords)
        self.isolation_score = calculate_knn_isolation(self.matrix, k=5)
        # Calculate distances from depot to clients and normalize row-wise to [0,1]
        self.dist_to_depot = self.matrix[:, 0, 0:]  # Distances from depot to clients
        # Find min and max values per batch for normalization
        min_dist = torch.min(self.dist_to_depot, dim=1, keepdim=True)[0]
        max_dist = torch.max(self.dist_to_depot, dim=1, keepdim=True)[0]
        # Avoid division by zero
        divisor = torch.clamp(max_dist - min_dist, min=1e-10)
        # Normalize each row to [0,1] range
        self.dist_to_depot = ((self.dist_to_depot - min_dist) / divisor).unsqueeze(-1)
        # coords of depot
        self.depot_coords = self.coords[:, 0, :].unsqueeze(1)  # [batch, 1, 2]
        # q / Q demand_normalized
        self.demand_normalized = (self.demands / self.capacity).unsqueeze(
            -1
        )  # [batch, num_nodes+1, 1]

    # --------------------------------
    # Problem Instance Generation
    # --------------------------------

    def generate_params(
        self,
        mode: str = "train",
        pb: bool = False,
        coords: torch.Tensor = torch.Tensor(),
        demands: torch.Tensor = torch.Tensor(),
        capacities: Union[int, torch.Tensor] = 30,
    ) -> None:
        """
        Generate problem instances with optional clustering.

        Args:
            mode: 'train' or 'test' (affects random seed)
            pb: Whether to load predefined problem
            coords: Optional coordinates tensor
            demands: Optional demands tensor

        Returns:
            Dictionary containing:
            - coords: Node coordinates [batch, num_nodes+1, 2]
            - demands: Node demands [batch, num_nodes+1] (depot demand=0)
        """
        if mode == "test":
            self.manual_seed(0)  # Fixed seed for reproducibility in test mode

        if isinstance(capacities, int):
            capacities = torch.full(
                (self.n_problems, 1),
                capacities,
                device=self.device,
                dtype=torch.float32,
            )

        if pb:  # Use provided problem data
            if coords.shape[0] != self.n_problems:
                raise ValueError(
                    f"Expected {self.n_problems} problems, got {coords.shape[0]}"
                )
            if demands.shape[0] != self.n_problems:
                raise ValueError(
                    f"Expected {self.n_problems} problems, got {demands.shape[0]}"
                )
            params = {"coords": coords, "demands": demands, "capacity": capacities}

        else:
            params = self._generate_random_instances()
            params["capacity"] = capacities

        self.set_params(params)

    def _generate_random_instances(self) -> Dict[str, torch.Tensor]:
        """
        Generate completely random problem instances.

        Coordinates are uniformly distributed in the unit square.

        Returns:
            Dictionary with coordinates and demands
        """
        coords = torch.rand(
            self.n_problems,
            self.dim + 1,
            2,
            device=self.device,
            generator=self.generator,
        )
        demands = self._generate_demands()
        return {"coords": coords, "demands": demands}

    def _generate_demands(self) -> torch.Tensor:
        """
        Generate random customer demands (depot demand=0).

        Returns:
            Tensor of demand values
        """
        demands = torch.randint(
            1,
            10,
            (self.n_problems, self.dim + 1),
            device=self.device,
            generator=self.generator,
        )
        demands[:, 0] = 0  # Depot has no demand
        return demands

    # --------------------------------
    # State and Solution Representation
    # --------------------------------

    def get_distance_to_centroid(self, solution: torch.Tensor) -> torch.Tensor:
        """
        Calculates the Euclidean distance between each node and the centroid
        (center of gravity) of the route it belongs to.

        Args:
            solution: [batch, seq_len, 1]

        Returns:
            [batch, seq_len, 1] Distance to centroid for each node
        """
        # 1. Get coordinates aligned with the solution sequence [B, L, 2]
        coords = self.get_coords(solution)

        # 2. Prepare tensors for scatter operations
        # We need to know the max segment ID to size our tensors correctly
        num_routes = self.segment_ids.max() + 1
        batch_size, seq_len = self.segment_ids.shape

        # Initialize Sums and Counts
        # shape: [B, num_routes, 2]
        route_coord_sums = torch.zeros(
            batch_size, int(num_routes), 2, device=self.device, dtype=coords.dtype
        )
        # shape: [B, num_routes, 1]
        route_node_counts = torch.zeros(
            batch_size, int(num_routes), 1, device=self.device, dtype=coords.dtype
        )

        # 3. Aggregate data per route (Scatter Add)
        # self.segment_ids must be expanded to [B, L, 2] for coords scatter
        segment_ids_expanded = self.segment_ids.unsqueeze(-1).expand(-1, -1, 2)

        # Sum X and Y for each route
        route_coord_sums.scatter_add_(1, segment_ids_expanded, coords)

        # Count nodes in each route (summing 1s)
        ones = torch.ones(
            batch_size, seq_len, 1, device=self.device, dtype=coords.dtype
        )
        # Only count actual nodes (where segment_id > 0), ignore depot padding if necessary
        # However, your segment_ids usually assigns 0 to depot/padding.
        # Ideally we only care about actual routes.
        # Using self.mask ensures we don't count padding/depot visits as "route nodes"
        # if your segment logic treats them as separate.
        # Assuming self.segment_ids identifies valid routes for clients:
        route_node_counts.scatter_add_(1, self.segment_ids.unsqueeze(-1), ones)

        # 4. Calculate Centroids
        # Avoid division by zero for empty/padding routes
        route_node_counts = torch.clamp(route_node_counts, min=1.0)
        route_centroids = route_coord_sums / route_node_counts  # [B, num_routes, 2]

        # 5. Map centroids back to node positions
        # Gather the centroid relevant to each node in the sequence
        # Shape becomes [B, L, 2] matching 'coords'
        node_centroids = route_centroids.gather(1, segment_ids_expanded)

        # 6. Calculate Euclidean Distance
        dists = (coords - node_centroids).norm(p=2, dim=-1, keepdim=True)

        return dists

    def set_feature_flags(self, feature_flags: Dict[str, bool]) -> None:
        """Store feature flags and pre-calculate input dimension."""
        self.feature_flags = feature_flags

    def get_input_dim(self) -> int:
        """Dynamically calculates the channel dimension 'c' based on active flags."""
        # Base dimensions for each group (Update these numbers based on your exact tensor sizes!)
        dims = {
            "static": 7,  # x, y, th, d, is_depot, knn, q/Q
            "topology": 4,  # prev_x, prev_y, next_x, next_y
            "detour": 1,  # detour
            "centroid": 1,  # centroid_dist
            "route_status": 4,  # load/Q, slack, q/load, route_cost_norm
            "route_cost": 1,  # cost of current route
            "route_pct": 1,  # capacity signal of the route
            "slack": 1,  # (1 - route_pct)
            "node_pct": 1,  # node demand percentage with route capacity
            "meta": 2,  # temp, progress
        }

        total_dim = 0
        for group, is_active in self.feature_flags.items():
            if is_active:
                total_dim += dims[group]
        return total_dim

    def build_state_components(self, x, temp, time):
        """
        Build state components for the model.

        Combines solution representation with problem features and metadata.

        Args:
            x: Solution tensor
            temp: Temperature parameter
            time: Time step information

        Returns:
            List of state component tensors
        """
        flags = self.feature_flags
        components = [x]
        padding = max(0, x.size(1) - self.state_encoding.size(1))
        padded_coords = F.pad(self.state_encoding, (0, 0, 0, padding)).gather(
            1, x.expand(-1, -1, 2)
        )

        # --- 1. STATIC ---
        if flags.get("static", True):
            is_depot = (x == 0).long()

            components.extend(
                [
                    padded_coords,  # coords (2)
                    is_depot,  # is_depot (1)
                    self.angles.gather(1, x),  # theta (1)
                    self.dist_to_depot.gather(1, x),  # dist (1)
                    self.demand_normalized.gather(1, x),  # Normalized demands
                    self.isolation_score.gather(1, x),  # Isolation score
                ]
            )
        # --- 2. TOPOLOGY ---
        if flags.get("topology", True):
            prev_coords = torch.roll(padded_coords, shifts=1, dims=1)
            next_coords = torch.roll(padded_coords, shifts=-1, dims=1)
            components.extend([prev_coords, next_coords])

        # --- 3. LOCAL COST ---
        if flags.get("detour", True):
            components.append(calculate_detour_features(x, self.matrix))

        if flags.get("centroid", False):
            components.append(self.get_distance_to_centroid(x))

        # --- 4. ROUTE STATUS ---
        # Pre-calculate only if needed to save compute
        if any(flags.get(k) for k in ["route_pct", "slack", "node_pct"]):
            node_pct, route_pct, remaining_capacity_fraction = (
                self.get_percentage_demands()
            )

            if flags.get("route_pct", True):
                components.append(route_pct)

            if flags.get("slack", False):
                components.append(remaining_capacity_fraction)

            if flags.get("node_pct", False):
                components.append(node_pct)

        if flags.get("route_cost", False):
            components.append(self.cost_per_route(x))
        # --- 5. META ---
        if flags.get("meta", True):
            components.extend([repeat_to(temp, x), repeat_to(time, x)])

        return components

    @property
    def state_encoding(self) -> torch.Tensor:
        """
        Get node coordinates as static problem features.

        Returns:
            Coordinates tensor
        """
        return self.coords

    def get_coords(self, solution: torch.Tensor) -> torch.Tensor:
        """
        Get coordinates in solution order.

        Args:
            solution: Solution tensor

        Returns:
            Ordered coordinates tensor
        """
        return torch.gather(
            self.coords, 1, solution.expand(-1, -1, self.coords.size(-1))
        )

    def get_demands(self, solution: torch.Tensor) -> torch.Tensor:
        """
        Get demands in solution order.

        Args:
            solution: Solution tensor

        Returns:
            Ordered demands tensor
        """
        return torch.gather(self.demands, 1, solution.squeeze(-1))

    def generate_init_state(
        self, init_heuristic: str = "", multi_init: bool = False
    ) -> torch.Tensor:
        """
        Generate initial solutions using specified algorithm or multiple methods
        if MULTI_INIT is enabled.
        """

        if multi_init:
            split_size = self.n_problems // len(self.params["INIT_LIST"])
            solutions = []
            for i, method in enumerate(self.params["INIT_LIST"]):
                sol = init_methods[method](self).to(self.device)
                if i == len(self.params["INIT_LIST"]) - 1:
                    solutions.append(sol[i * split_size :, :, :])
                else:
                    solutions.append(sol[i * split_size : (i + 1) * split_size, :, :])
            max_size = max(sol.shape[1] for sol in solutions)
            solutions_padded = [
                F.pad(sol, (0, 0, 0, max_size - sol.shape[1])) for sol in solutions
            ]
            sol = torch.cat(solutions_padded, dim=0)
        else:
            if init_heuristic not in init_methods:
                raise ValueError(f"Unsupported initialization method: {init_heuristic}")

            sol = init_methods[init_heuristic](self).to(self.device)
        self.ordered_demands = self.get_demands(sol)
        valid = is_feasible(sol, self.ordered_demands, self.capacity).all()
        if not valid:
            raise ValueError("Generated initial solution is not feasible.")
        # Identify route segments
        self.mask = self.ordered_demands != 0
        segment_start = self.mask & ~torch.cat(
            [torch.zeros_like(self.mask[:, :1]), self.mask[:, :-1]], dim=1
        )
        # Compute segment demands
        self.segment_ids = torch.cumsum(segment_start, 1) * self.mask

        return sol

    # --------------------------------
    # Cost Calculation
    # --------------------------------

    def cost(self, solution: torch.Tensor) -> torch.Tensor:
        """
        Compute total route length for given solution.

        Args:
            solution: Tensor [batch, route_length, 1] representing routes

        Returns:
            Tensor [batch] containing total distance for each solution
        """
        return torch.sum(self.get_edge_lengths_in_tour(solution), -1)

    def cost_per_route(self, solution: torch.Tensor) -> torch.Tensor:
        """
        Compute normalized cost contribution per route segment.

        Args:
            solution: Tensor [batch, route_length, 1] representing routes

        Returns:
            Tensor [batch, route_length, 1] with normalized route costs
        """
        edge_lengths = self.get_edge_lengths_in_tour(solution)
        total_cost = torch.sum(edge_lengths, -1, keepdim=True)

        segment_sums = torch.zeros_like(edge_lengths)
        segment_sums.scatter_add_(1, self.segment_ids, edge_lengths)
        route_costs = segment_sums.gather(1, self.segment_ids) * self.mask

        num_routes = self.segment_ids.max(dim=1, keepdim=True)[0]

        return (route_costs * (num_routes / total_cost)).unsqueeze(-1)

    def get_edge_lengths_in_tour(self, solution: torch.Tensor) -> torch.Tensor:
        """
        Compute Euclidean distances between consecutive nodes in solution.

        Args:
            solution: Tensor [batch, num_nodes, 1]

        Returns:
            Tensor [batch, num_nodes] of inter-node distances
        """
        coords = self.get_coords(solution)
        next_coords = torch.cat([coords[:, 1:, :], coords[:, :1, :]], dim=1)
        return (coords - next_coords).norm(p=2, dim=-1)

    # --------------------------------
    # Demand and Capacity Analysis
    # --------------------------------

    def get_percentage_demands(
        self,
    ) -> Tuple[torch.Tensor, ...]:
        """
        Compute demand-related percentages:
        1. Node demand as a fraction of route demand
        represents the proportion of a node's demand relative to the total demand
        of the route.
        2. Route demand as a fraction of vehicle capacity
        represents how much the route is loaded relative to the vehicle's capacity.
        3. Node demand as a fraction of vehicle capacity
        represents the proportion of the node's demand relative to the vehicle's
        capacity.

        Returns:
            Tuple of three percentage tensors
        """
        node_demands = self.ordered_demands
        total_demand_per_route = torch.zeros_like(node_demands)
        total_demand_per_route.scatter_add_(1, self.segment_ids, node_demands)
        route_demand_for_nodes = total_demand_per_route.gather(1, self.segment_ids)
        # q / route Load
        node_demand_fraction_of_route = torch.nan_to_num(
            node_demands / route_demand_for_nodes, nan=0.0
        )
        # route Load / Q
        route_demand_fraction_of_capacity = route_demand_for_nodes / self.capacity

        # (Q - route Load) / Q
        remaining_capacity_fraction = (
            self.capacity - route_demand_for_nodes
        ) / self.capacity

        return (
            node_demand_fraction_of_route.unsqueeze(-1),
            route_demand_fraction_of_capacity.unsqueeze(-1),
            remaining_capacity_fraction.unsqueeze(-1),
        )

    # --------------------------------
    # Solution Modification Heuristics
    # --------------------------------

    def update(
        self, solution: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Modify solution by applying heuristic action.

        Args:
            solution: Current solution tensor [batch, route_length, 1]
            action: Indices to modify [batch, 2] or [batch, 3] for mixed heuristic

        Returns:
            Tuple containing:
            - New solution after applying heuristic (or original if infeasible)
            - Validity flag for each solution in batch
        """
        # Apply the selected update method
        if self.params["UPDATE_METHOD"] == "rm_depot":
            # First approach: Remove depot visits to simplify operations
            # 1. Create mask identifying non-depot nodes
            mask = solution.squeeze(-1) != 0

            # 2. Extract only client nodes, reshaping to maintain batch dimension
            compact_sol = solution[mask].view(solution.size(0), -1, solution.size(-1))

            # 3. Apply heuristic on client-only solution
            modified_sol = self.apply_heuristic(compact_sol, action).long()

            # 4. Rebuild valid CVRP solution with depot visits inserted where needed
            sol = construct_cvrp_solution(modified_sol, self.demands, self.capacity)

            # Add padding to match the shape of the original solution
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

        else:
            # Second approach: Apply heuristic directly on full solution
            # (including depot visits)
            sol = self.apply_heuristic(solution, action).long()
            # Note: This approach may require post-processing to ensure feasibility

        new_ordered_demands = self.get_demands(sol)
        # Check if the modified solutions are feasible (respect capacity constraints)
        valid = (
            is_feasible(sol, new_ordered_demands, self.capacity).unsqueeze(-1).long()
        )

        # Return original solution if modified solution is infeasible
        sol = torch.where(valid.unsqueeze(-1) == 1, sol, solution).to(torch.int64)

        return sol, valid

    def update_tensor(self, solution: torch.Tensor):
        """
        Update internal tensors based on current solution.
        Args:
            solution: The current solution, shape [batch, num_nodes, 1].
        """
        # Update internal tensors based on new solution
        self.ordered_demands = self.get_demands(solution)
        # Identify route segments
        self.mask = self.ordered_demands != 0
        segment_start = self.mask & ~torch.cat(
            [torch.zeros_like(self.mask[:, :1]), self.mask[:, :-1]], dim=1
        )

        # Compute segment demands
        self.segment_ids = torch.cumsum(segment_start, 1) * self.mask

    def _get_current_route_loads(self) -> torch.Tensor:
        """
        Calculate the total load (sum of demands) for each route.

        Returns:
            A tensor [batch, max_routes] containing the load of each route.
            The index corresponds to the route ID (segment_id).
        """
        # Find the maximum number of routes in the batch to size the tensor
        num_routes = self.segment_ids.max() + 1
        route_loads = torch.zeros(
            self.n_problems,
            int(num_routes.item()),
            device=self.device,
            dtype=self.ordered_demands.dtype,
        )

        # scatter_add_ sums the demands (self.ordered_demands)
        # into the correct "bins" of routes (indexed by self.segment_ids)
        route_loads.scatter_add_(1, self.segment_ids, self.ordered_demands)
        return route_loads

    def get_action_mask(
        self, solution: torch.Tensor, node_pos: torch.Tensor
    ) -> torch.Tensor:
        """
        Generates a mask of valid actions for a given node, respecting
        vehicle capacity constraints.

        Args:
            solution: The current solution, shape [batch, num_nodes, 1].
            node_pos: The *position* index of the source node in the tour,
                    shape [batch, 1].

        Returns:
            A boolean mask of shape [batch, num_nodes] where `True`
            represents a valid action.
        """
        batch_size, num_nodes, _ = solution.shape

        # 1. Calculate current loads for all routes
        current_route_loads = self._get_current_route_loads()

        node_pos = node_pos.unsqueeze(-1)

        # 2. Get info for the source node (the one to be moved)
        # .gather() selects values at the indices specified by node_pos
        source_demand = torch.gather(self.ordered_demands, 1, node_pos)
        source_route_id = torch.gather(self.segment_ids, 1, node_pos)
        source_route_load = torch.gather(current_route_loads, 1, source_route_id)

        # 3. Get info for all potential target nodes
        target_demands = self.ordered_demands
        target_route_ids = self.segment_ids
        # .gather() here retrieves the route load for EACH target node
        target_route_loads = torch.gather(current_route_loads, 1, target_route_ids)

        # Mask for depot positions (depot has segment_id == 0)
        is_depot_mask = self.segment_ids == 0

        # --- Heuristic-specific logic ---
        heuristic_func = self.heuristic
        if heuristic_func is None:
            raise ValueError("Single heuristic is not configured.")

        # By default, apply the base topological mask (cannot act on oneself)
        mask = torch.ones(batch_size, num_nodes, device=self.device, dtype=torch.bool)
        # mask.scatter_(1, node_pos.long(), False)

        if heuristic_func is swap:
            # For a 'swap' between source node (A) and target node (B):
            # Load Route A' = Load Route A - Demand A + Demand B
            # Load Route B' = Load Route B - Demand B + Demand A

            # Calculate potential new loads for source and target routes
            new_load_for_source_route = (
                source_route_load - source_demand + target_demands
            )
            new_load_for_target_route = (
                target_route_loads - target_demands + source_demand
            )

            # Check validity
            source_route_ok = new_load_for_source_route <= self.capacity
            target_route_ok = new_load_for_target_route <= self.capacity

            # If the swap is intra-route, the total load doesn't change,
            # so it's always valid
            is_intra_route = source_route_id == target_route_ids

            # An action is valid if (it's intra-route) OR
            # (both new routes respect capacity)
            capacity_mask = torch.where(
                is_intra_route, torch.ones_like(mask), source_route_ok & target_route_ok
            )
            mask &= capacity_mask

            # Cannot swap with the depot
            mask &= ~is_depot_mask

        elif heuristic_func is insertion:
            # For an 'insertion' of source node (A) into the route of target node (B):
            # Route A is lightened (always valid).
            # Route B is heavier: Load Route B' = Load Route B + Demand A

            # This block handles a specific edge case where target_route_loads
            # might be 0. It tries to get a valid load by looking at the next node.
            mask = torch.ones_like(target_route_loads, dtype=torch.bool)
            batch_idx = torch.arange(target_route_loads.shape[0]).unsqueeze(-1)
            mask[batch_idx, node_pos] = False
            output_flat = target_route_loads[mask]
            remaining_nodes = output_flat.reshape(target_route_loads.shape[0], -1)

            target_route_loads = torch.cat(
                [
                    remaining_nodes,
                    torch.zeros(
                        batch_size,
                        1,
                        device=self.device,
                        dtype=target_route_loads.dtype,
                    ),
                ],
                dim=1,
            )
            shifted_target_loads = torch.cat(
                [
                    target_route_loads[:, 1].unsqueeze(-1),
                    target_route_loads[:, :-1],
                ],
                dim=1,
            )

            adjusted_target_loads = torch.max(target_route_loads, shifted_target_loads)

            new_load_for_target_route = adjusted_target_loads + source_demand
            capacity_mask = new_load_for_target_route <= self.capacity

            is_intra_route = source_route_id == target_route_ids

            # Valid if (it's intra-route) OR
            # (the new target route respects capacity)
            capacity_mask = torch.where(
                is_intra_route, torch.ones_like(mask), capacity_mask
            )
            mask &= capacity_mask

            # Ensure the last and first value of each row is False
            mask[:, -1] = False
            mask[:, 0] = False

        elif heuristic_func is two_opt:
            raise NotImplementedError(
                "2-opt is not implemented for UPDATE_METHOD == 'two_opt'."
            )

        else:
            raise NotImplementedError(
                f"Masking for heuristic {heuristic_func.__name__} is not implemented."
            )

        # --- Final invalidation logic ---

        # If source_demand, source_route_id, or source_route_load is 0,
        # the source node is invalid (e.g., it's a depot or padding)
        invalid_source = (
            (source_demand == 0) | (source_route_id == 0) | (source_route_load == 0)
        )

        # If the entire mask row is False (no valid moves found),
        # treat the source as invalid as well.
        invalid_source |= ~mask.any(dim=1, keepdim=True)

        # Create a mask that is only `True` at the source node's position.
        # This is used as a "no-op" or "sentinel" action.
        empty_mask = torch.zeros_like(mask, dtype=torch.bool)
        source_only_mask = empty_mask.scatter(1, node_pos.long(), True)

        # If the source was invalid, replace the calculated mask with the
        # 'source_only_mask'. Otherwise, keep the calculated mask.
        mask = torch.where(invalid_source, source_only_mask, mask)

        return mask
