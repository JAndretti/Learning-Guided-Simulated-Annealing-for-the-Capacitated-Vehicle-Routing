from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from problem import CVRP
from utils import repeat_to


def create_network(input_dim, embed_dim, num_hidden_layers, device):
    layers = []
    # Entry layer
    layers.append(nn.Linear(input_dim, embed_dim, bias=True, device=device))
    layers.append(nn.LeakyReLU())

    # Hidden layers
    for _ in range(num_hidden_layers):
        layers.append(nn.Linear(embed_dim, embed_dim, bias=True, device=device))
        layers.append(nn.LeakyReLU())

    # Output layer
    layers.append(nn.Linear(embed_dim, 1, bias=False, device=device))

    return nn.Sequential(*layers).to(device)


class SAModel(nn.Module):
    def __init__(self, device: str = "cpu") -> None:
        super().__init__()
        self.device = device
        self.generator = torch.Generator(device=device)

    def manual_seed(self, seed: int) -> None:
        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(seed)

    def get_logits(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def sample(
        self, state: torch.Tensor, greedy: bool = False, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def baseline_sample(
        self, state: torch.Tensor, **kwargs
    ) -> Tuple[torch.Tensor, None, torch.Tensor]:
        # Output must match sample output
        raise NotImplementedError

    @staticmethod
    def init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            # Orthogonal init is the gold standard for PPO
            nn.init.orthogonal_(m.weight, gain=np.sqrt(2))

            # Biases are usually set to 0 rather than 0.01
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)


class CVRPActorPairs(SAModel):
    """Actor network for CVRP that selects pairs of nodes to swap."""

    def __init__(
        self,
        embed_dim: int = 32,
        c: int = 22,
        num_hidden_layers: int = 2,
        device: str = "cpu",
        mixed_heuristic: bool = False,
        method: str = "free",
    ) -> None:
        super().__init__(device)
        self.mixed_heuristic = mixed_heuristic
        self.method = method

        self.input_dim = c * 2 if mixed_heuristic else c * 2 - 2
        self.net = create_network(
            self.input_dim,
            embed_dim,
            num_hidden_layers=num_hidden_layers,
            device=device,
        )

        self.net.apply(self.init_weights)

    def forward(self, state):
        """Forward pass computing logits for node pairs."""
        pair_logits = self.net(state)
        return pair_logits

    @staticmethod
    def init_weights(m: nn.Module) -> None:
        """Initialize weights using Kaiming uniform initialization."""
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight)
            if m.bias is not None:
                m.bias.data.fill_(0.01)

    def sample_from_logits(
        self, logits: torch.Tensor, greedy: bool = False, one_hot: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample actions from logits using either greedy or multinomial sampling."""
        probs = torch.softmax(logits, dim=-1)

        if greedy:
            smpl = torch.argmax(probs, -1, keepdim=False)
        else:
            smpl = torch.multinomial(probs, 1, generator=self.generator)[..., 0]

        taken_probs = probs.gather(1, smpl.view(-1, 1))

        if one_hot:
            smpl = F.one_hot(smpl, num_classes=logits.shape[-1])[..., None]

        return smpl, torch.log(taken_probs)

    def get_logits(
        self, state: torch.Tensor, action: torch.Tensor, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute logits and log probabilities for given state and action."""
        pair_features, idx1, idx2, heuristic_indices = self._prepare_features_and_pairs(
            state
        )

        # Forward pass
        outputs = self.forward(pair_features)
        logits = outputs[..., 0]
        probs = torch.softmax(logits, dim=-1)
        log_probs = torch.log(probs)
        all_action = (
            (
                torch.cat(
                    [
                        idx1.unsqueeze(-1).repeat_interleave(2, dim=0),
                        idx2.unsqueeze(-1).repeat_interleave(2, dim=0),
                        heuristic_indices.unsqueeze(-1),
                    ],
                    dim=-1,
                )
                if heuristic_indices is not None
                else torch.cat([idx1.unsqueeze(-1), idx2.unsqueeze(-1)], dim=-1)
            ),
        )

        return log_probs, all_action

    def baseline_sample(
        self, state: torch.Tensor, **kwargs
    ) -> Tuple[torch.Tensor, None]:
        """Generate baseline sample using uniform probabilities."""
        n_problems, problem_dim, dim = state.shape
        state = state[:, :, :-2]  # Remove temperature and time

        x, _, *_ = torch.split(state, [1, 2] + [1] * (dim - 5), dim=-1)
        mask = x.squeeze(-1) != 0
        x = torch.stack([c[m] for c, m in zip(x, mask)], dim=0)

        idx1, idx2 = torch.triu_indices(x.shape[1], x.shape[1], offset=1)
        logits = torch.ones(n_problems, idx1.shape[0]).to(self.device)
        c, _ = self.sample_from_logits(logits, one_hot=False)

        if self.mixed_heuristic:
            pair_idx = c // 2
            heuristic_idx = c % 2
            action = torch.stack(
                [idx1[pair_idx], idx2[pair_idx], heuristic_idx], dim=-1
            )
        else:
            action = torch.stack([idx1[c], idx2[c]], dim=-1)

        return action, None

    def sample(
        self, state: torch.Tensor, greedy: bool = False, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample an action pair from the current state."""
        step = 1000
        if state.shape[0] * state.shape[1] > 1000 * 161:
            actions = []
            log_probs_list = []
            for i in range(0, state.shape[0], step):
                chunk = state[i : i + step]
                pair_features, idx1, idx2 = self._prepare_features_and_pairs(chunk)
                logits = self.forward(pair_features)[..., 0]  # Forward pass

                c, log_probs = self.sample_from_logits(
                    logits, greedy=greedy, one_hot=False
                )

                if self.mixed_heuristic:
                    pair_idx = c // 2
                    heuristic_idx = c % 2
                    action = torch.stack(
                        [idx1[pair_idx], idx2[pair_idx], heuristic_idx], dim=-1
                    )
                else:
                    action = torch.stack([idx1[c], idx2[c]], dim=-1)

                actions.append(action)
                log_probs_list.append(log_probs)

                # explicitly free memory
                del pair_features, idx1, idx2, logits, c
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # Stack results from all chunks
            action = torch.cat(actions, dim=0)
            log_probs = torch.cat(log_probs_list, dim=0)
        else:
            pair_features, idx1, idx2 = self._prepare_features_and_pairs(state)
            logits = self.forward(pair_features)[..., 0]  # Forward pass

            c, log_probs = self.sample_from_logits(logits, greedy=greedy, one_hot=False)

            if self.mixed_heuristic:
                pair_idx = c // 2
                heuristic_idx = c % 2
                action = torch.stack(
                    [idx1[pair_idx], idx2[pair_idx], heuristic_idx], dim=-1
                )
            else:
                action = torch.stack([idx1[c], idx2[c]], dim=-1)
        mask = torch.ones_like(logits, dtype=torch.bool)
        return action, log_probs[..., 0], mask

    def evaluate(
        self, state: torch.Tensor, action: torch.Tensor, mask: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        """Evaluate log probabilities of given actions."""
        pair_features, idx1, idx2 = self._prepare_features_and_pairs(state)
        if self.mixed_heuristic:
            # Find the pair index
            pair_mask = (idx1 == action[:, 0].unsqueeze(1)) & (
                idx2 == action[:, 1].unsqueeze(1)
            )
            pair_idx = pair_mask.nonzero(as_tuple=True)[1]
            # Calculate the full action index (pair_idx * 2 + heuristic_idx)
            action_idx = pair_idx * 2 + action[:, 2]
        else:
            action_idx = (idx1 == action[:, 0].unsqueeze(1)) & (
                idx2 == action[:, 1].unsqueeze(1)
            )
            action_idx = action_idx.nonzero(as_tuple=True)[1]

        # Forward pass
        step = 1000
        if state.shape[0] * state.shape[1] > 1000 * 161:
            logits_list = []
            for i in range(0, state.shape[0], step):
                chunk = state[i : i + step]
                pair_features, idx1, idx2 = self._prepare_features_and_pairs(chunk)
                logits = self.forward(pair_features)[..., 0]  # Forward pass
                logits_list.append(logits)
                del pair_features, logits
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            logits = torch.cat(logits_list, dim=0)
        else:
            logits = self.forward(pair_features)[..., 0]

        # Compute action probabilities
        probs = torch.softmax(logits, dim=-1)
        log_probs = torch.log(probs.gather(1, action_idx.view(-1, 1)))
        return log_probs[..., 0]

    def _prepare_features_and_pairs(
        self, state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Helper method to prepare features and pairs from state."""
        n_problems, problem_dim, dim = state.shape
        temp, time = state[:, :, -2], state[:, :, -1]
        state = state[:, :, :-2]  # Remove temperature and time

        x, coords, *extra_features = torch.split(
            state, [1, 2] + [1] * (dim - 5), dim=-1
        )

        # Gather coordinate information
        coords = coords.gather(1, x.long().expand_as(coords))
        coords_prev = torch.cat([coords[:, -1:, :], coords[:, :-1, :]], dim=1)
        coords_next = torch.cat([coords[:, 1:, :], coords[:, :1, :]], dim=1)

        c_state = torch.cat([coords, coords_prev, coords_next] + extra_features, -1)

        if self.method == "rm_depot":
            mask = x.squeeze(-1) != 0
            c_state = c_state[mask].view(n_problems, -1, c_state.size(-1))

        # Get all possible pairs
        idx1, idx2 = torch.triu_indices(
            c_state.shape[1], c_state.shape[1], offset=1, device=c_state.device
        )
        x_pairs_1 = c_state[:, idx1, :]
        x_pairs_2 = c_state[:, idx2, :]

        # Combine pair features with temperature and time
        pair_features = torch.cat(
            [
                x_pairs_1,
                x_pairs_2,
                temp[:, idx1].unsqueeze(-1),
                time[:, idx1].unsqueeze(-1),
            ],
            dim=-1,
        )
        if self.mixed_heuristic:
            # Duplicate each pair for both heuristic options
            n_pairs = pair_features.shape[1]
            pair_features = pair_features.repeat_interleave(2, dim=1)

            # Add one-hot encoding for heuristic selection
            heuristic_indices = torch.arange(2, device=c_state.device).repeat(n_pairs)
            heuristic_one_hot = F.one_hot(heuristic_indices, num_classes=2).repeat(
                n_problems, 1, 1
            )

            # Concatenate the one-hot encoding to the features
            pair_features = torch.cat([pair_features, heuristic_one_hot], dim=-1)

        return pair_features, idx1, idx2


class CVRPActor(SAModel):
    def __init__(
        self,
        embed_dim: int = 32,
        c: int = 13,
        num_hidden_layers: int = 2,
        device: str = "cpu",
        mixed_heuristic: bool = False,
        method: str = "free",
    ) -> None:
        super().__init__(device)
        self.mixed_heuristic = mixed_heuristic
        self.c1_state_dim = c
        self.method = method
        self.c2_state_dim = c * 2 if mixed_heuristic else c * 2 - 2

        self.city1_net = create_network(
            self.c1_state_dim,
            embed_dim,
            num_hidden_layers=num_hidden_layers,
            device=device,
        )

        self.city2_net = create_network(
            self.c2_state_dim,
            embed_dim,
            num_hidden_layers=num_hidden_layers,
            device=device,
        )

        # Apply the generic PPO init to the WHOLE network
        self.city1_net.apply(self.init_weights)

        # 3. Manually overwrite the Output Layer (The last layer in the list)
        # We access it using [-1] because it is the last item added to layers[]
        last_layer = self.city1_net[-1]

        # IF THIS IS AN ACTOR (Policy):
        # We use 0.01 so actions start out random/uniform (Crucial for PPO)
        nn.init.orthogonal_(last_layer.weight, gain=0.01)

        # Ensure bias is 0 for the output
        if last_layer.bias is not None:
            nn.init.constant_(last_layer.bias, 0.0)

        self.city2_net.apply(self.init_weights)
        last_layer = self.city2_net[-1]
        nn.init.orthogonal_(last_layer.weight, gain=0.01)
        # Ensure bias is 0 for the output
        if last_layer.bias is not None:
            nn.init.constant_(last_layer.bias, 0.0)

    def sample_from_logits(
        self, logits: torch.Tensor, greedy: bool = False, one_hot: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample actions from logits using either greedy or multinomial sampling."""
        n_problems, problem_dim = logits.shape
        probs = torch.softmax(logits, dim=-1)

        if greedy:
            smpl = torch.argmax(probs, -1, keepdim=False)
        else:
            smpl = torch.multinomial(probs, 1, generator=self.generator)[..., 0]

        taken_probs = probs.gather(1, smpl.view(-1, 1))

        if one_hot:
            smpl = F.one_hot(smpl, num_classes=problem_dim)[..., None]

        return smpl, torch.log(taken_probs)

    def get_logits(
        self, state: torch.Tensor, action: torch.Tensor, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute logits and log probabilities for given state and action."""
        c1_state, n_problems, routes_ids = self._prepare_features_city1(state)

        c1 = action[:, 0]

        # City 1 net
        logits = self.city1_net(c1_state)[..., 0]
        probs = torch.softmax(logits, dim=-1)
        log_probs_c1 = torch.log(probs)

        c2_state = self._prepare_features_city2(
            c1_state, c1, n_problems
        )  # Second city encoding
        logits = self.city2_net(c2_state)[..., 0]

        probs = torch.softmax(logits, dim=-1)
        log_probs_c2 = torch.log(probs)

        return log_probs_c1, log_probs_c2

    def baseline_sample(
        self, state: torch.Tensor, problem=CVRP, **kwargs
    ) -> Tuple[torch.Tensor, None, torch.Tensor]:
        """Generate baseline sample using uniform probabilities."""
        n_problems, problem_dim, _ = state.shape
        x = state[:, :, 0]
        mask = x.squeeze(-1) != 0
        # Sample c1 at random
        if self.method == "rm_depot":
            x = x[mask].view(n_problems, -1)
            logits = torch.ones(n_problems, x.shape[1]).to(self.generator.device)
        else:
            logits = torch.ones(n_problems, x.shape[1]).to(self.generator.device)
            logits[~mask] = -float("inf")  # Mask logits where x == 0
        c1, _ = self.sample_from_logits(logits, one_hot=False)

        # sample c2
        if self.mixed_heuristic:
            logits = torch.ones(n_problems, x.shape[1] * 2).to(self.generator.device)
            c2, _ = self.sample_from_logits(logits, one_hot=False)
            heuristic_idx = c2 % 2
            action = torch.cat(
                [
                    c1.view(-1, 1).long(),
                    c2.view(-1, 1).long(),
                    heuristic_idx.view(-1, 1).long(),
                ],
                dim=-1,
            )
        else:
            logits = torch.ones(n_problems, x.shape[1]).to(self.generator.device)
            mask = torch.ones_like(logits, dtype=torch.bool)
            if self.method == "valid":
                mask = problem.get_action_mask(x.unsqueeze(-1), c1)
                logits[~mask] = -float("inf")  # Mask invalid actions
            else:
                arange = torch.arange(n_problems).to(logits.device)
                logits[arange, c1] = -float("inf")
            c2, _ = self.sample_from_logits(logits, one_hot=False)
            action = torch.cat([c1.view(-1, 1).long(), c2.view(-1, 1).long()], dim=-1)
        return action, None, mask

    def sample(
        self, state: torch.Tensor, greedy: bool = False, problem=CVRP, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample an action pair from the current state."""
        c1_state, n_problems, x = self._prepare_features_city1(state)

        logits = self.city1_net(c1_state)[..., 0]
        if self.method != "rm_depot":
            mask = (x != 0).squeeze(-1)
            logits[~mask] = -float("inf")  # Mask logits where x == 0

        c1, log_probs_c1 = self.sample_from_logits(logits, greedy=greedy, one_hot=False)

        c2_state = self._prepare_features_city2(
            c1_state, c1, n_problems
        )  # Second city encoding

        # City 2 net
        logits = self.city2_net(c2_state)[..., 0]
        mask = torch.ones_like(logits, dtype=torch.bool)
        if self.method == "valid":
            mask = problem.get_action_mask(solution=x, node_pos=c1)
            logits[~mask] = -float("inf")  # Mask invalid actions
        else:
            arange = torch.arange(n_problems).to(logits.device)
            logits[arange, c1] = -float("inf")

        c2, log_probs_c2 = self.sample_from_logits(logits, greedy=greedy, one_hot=False)

        # Construct action and log-probabilities
        if self.mixed_heuristic:
            c2_idx = c2 // 2  # Index of the pair
            heuristic_idx = c2 % 2  # Index of the heuristic
            action = torch.cat(
                [
                    c1.view(-1, 1).long(),
                    c2_idx.view(-1, 1).long(),
                    heuristic_idx.view(-1, 1).long(),
                ],
                dim=-1,
            )
        else:
            # Concatenate c1 and c2
            action = torch.cat([c1.view(-1, 1).long(), c2.view(-1, 1).long()], dim=-1)

        log_probs = log_probs_c1 + log_probs_c2
        return action, log_probs[..., 0], mask

    def evaluate(
        self, state: torch.Tensor, action: torch.Tensor, mask: torch.Tensor, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate actions to get their Log-Probabilities and the Entropy of the distribution.

        Args:
            state: Tensor of shape (batch, problem_size, features)
            action: Tensor of shape (batch, 2) containing [City1_Index, City2_Index]
            mask: Tensor indicating valid choices for the second city

        Returns:
            log_probs: The log-likelihood of the specific actions taken.
            total_entropy: The entropy (uncertainty) of the entire policy distribution.
        """

        # =========================================================================
        # PART 1: PREPARE DATA
        # =========================================================================
        c1_state, n_problems, x = self._prepare_features_city1(state)

        # Extract the specific actions the agent took previously
        # We need these to calculate how "likely" those specific moves were.
        taken_c1 = action[:, 0]

        if self.mixed_heuristic:
            # Handle special mixed heuristic action space
            c2_idx = action[:, 1]
            heuristic_idx = action[:, 2]
            taken_c2 = c2_idx * 2 + heuristic_idx
        else:
            taken_c2 = action[:, 1]

        # =========================================================================
        # PART 2: CITY 1 (First Decision)
        # =========================================================================

        # 1. Forward Pass: Get raw scores (logits) for every possible city
        logits_c1 = self.city1_net(c1_state)[..., 0]

        # 2. Masking: Determine which cities are valid to visit
        # We start by assuming all are valid, then filter based on the problem state
        valid_mask_c1 = torch.ones_like(logits_c1, dtype=torch.bool)

        if self.method != "rm_depot":
            # x != 0 usually implies we are not at the depot or the node is unvisited
            tmp_mask = (x != 0).squeeze(-1)
            # Set invalid actions to negative infinity so Softmax makes them 0.0
            logits_c1[~tmp_mask] = -float("inf")
            valid_mask_c1 = tmp_mask

        # 3. Probabilities: Convert scores to probabilities (0.0 to 1.0)
        # We compute log_softmax for numerical stability in loss calculations
        probs_c1 = torch.softmax(logits_c1, dim=-1)
        log_probs_all_c1 = torch.log_softmax(logits_c1, dim=-1)

        # 4. Entropy Calculation (Uncertainty Metric)
        # Formula: - Sum( p * log(p) )
        # We must use masking to avoid NaN errors (0.0 * -inf = NaN)
        p_log_p_c1 = torch.zeros_like(probs_c1)
        p_log_p_c1[valid_mask_c1] = (
            probs_c1[valid_mask_c1] * log_probs_all_c1[valid_mask_c1]
        )
        entropy_c1 = -p_log_p_c1.sum(dim=-1)

        # 5. Retrieve Log-Prob for the SPECIFIC action taken
        # .gather() picks the log-prob corresponding to the index in 'taken_c1'
        chosen_log_prob_c1 = log_probs_all_c1.gather(1, taken_c1.view(-1, 1)).squeeze(
            -1
        )

        # =========================================================================
        # PART 3: CITY 2 (Second Decision)
        # =========================================================================

        # Prepare state for the second decision (conditioned on the first choice)
        c2_state = self._prepare_features_city2(c1_state, taken_c1, n_problems)

        # 1. Forward Pass
        logits_c2 = self.city2_net(c2_state)[..., 0]

        # 2. Masking
        valid_mask_c2 = torch.ones_like(logits_c2, dtype=torch.bool)

        if self.method == "valid":
            # Use the pre-computed mask passed from outside
            logits_c2[~mask] = -float("inf")
            valid_mask_c2 = mask
        else:
            # Simple masking: Just prevent picking the same city again (taken_c1)
            arange = torch.arange(n_problems, device=logits_c2.device)
            logits_c2[arange, taken_c1] = -float("inf")
            valid_mask_c2[arange, taken_c1] = False

        # 3. Probabilities
        probs_c2 = torch.softmax(logits_c2, dim=-1)
        log_probs_all_c2 = torch.log_softmax(logits_c2, dim=-1)

        # 4. Entropy Calculation
        p_log_p_c2 = torch.zeros_like(probs_c2)
        p_log_p_c2[valid_mask_c2] = (
            probs_c2[valid_mask_c2] * log_probs_all_c2[valid_mask_c2]
        )
        entropy_c2 = -p_log_p_c2.sum(dim=-1)

        # 5. Retrieve Log-Prob for the SPECIFIC action taken
        chosen_log_prob_c2 = log_probs_all_c2.gather(1, taken_c2.view(-1, 1)).squeeze(
            -1
        )

        # =========================================================================
        # PART 4: COMBINE AND RETURN
        # =========================================================================

        # Total Log Likelihood of the path: log(P(c1)) + log(P(c2))
        total_log_probs = chosen_log_prob_c1 + chosen_log_prob_c2

        # Total Entropy of the policy: H(c1) + H(c2)
        total_entropy = entropy_c1 + entropy_c2

        return total_log_probs, total_entropy

    def _prepare_features_city1(
        self, state: torch.Tensor
    ) -> Tuple[torch.Tensor, int, torch.Tensor]:
        """Helper method to prepare features from state."""
        n_problems, problem_dim, dim = state.shape

        x = state[:, :, :1]
        c_state = state[:, :, 1:]

        if self.method == "rm_depot":
            mask = x.squeeze(-1) != 0
            c_state = c_state[mask].view(n_problems, -1, c_state.size(-1))
        return c_state, n_problems, x

    def _prepare_features_city2(
        self, c1_state: torch.Tensor, c1: torch.Tensor, n_problems: int
    ) -> torch.Tensor:
        """Helper method to prepare features from state."""
        # Second city encoding
        arange = torch.arange(n_problems)
        c1_val = c1_state[arange, c1]
        base = torch.cat([c1_val], -1)[:, None, :]

        base = repeat_to(base, c1_state)
        c1_state = c1_state[:, :, :-2]
        c2_state = torch.cat([base, c1_state], -1)
        if self.mixed_heuristic:
            # Duplicate each pair for both heuristic options
            n_row = c2_state.shape[1]
            c2_state = c2_state.repeat_interleave(2, dim=1)

            # Add one-hot encoding for heuristic selection
            heuristic_indices = torch.arange(2, device=c1_state.device).repeat(n_row)
            heuristic_one_hot = F.one_hot(heuristic_indices, num_classes=2).repeat(
                n_problems, 1, 1
            )
            # Concatenate the one-hot encoding to the features
            c2_state = torch.cat([c2_state, heuristic_one_hot], dim=-1)
        return c2_state


class CVRPCritic(nn.Module):
    """Critic network for CVRP that estimates state values."""

    def __init__(
        self,
        embed_dim: int,
        c: int = 13,
        num_hidden_layers: int = 2,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        self.q_func = create_network(
            c,
            embed_dim,
            num_hidden_layers=num_hidden_layers,
            device=device,
        )
        self.q_func.apply(self.init_weights)
        last_layer = self.q_func[-1]
        # IF THIS IS A CRITIC (Value Function):
        # We use 1.0 because the value estimate shouldn't be squashed too small
        nn.init.orthogonal_(last_layer.weight, gain=1.0)
        # Ensure bias is 0 for the output
        if last_layer.bias is not None:
            nn.init.constant_(last_layer.bias, 0.0)

    @staticmethod
    def init_weights(m: nn.Module) -> None:
        """Initialize weights using Kaiming uniform initialization."""
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight)
            if m.bias is not None:
                m.bias.data.fill_(0.01)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass computing state values."""
        n_problems, problem_dim, dim = state.shape
        state = state[:, :, 1:]

        q_values = self.q_func(state).view(n_problems, problem_dim)

        q_values = q_values.mean(dim=-1)
        return q_values


class CVRPCriticAttention(nn.Module):
    """
    Critic network for CVRP that estimates state values using Attention Pooling.

    This architecture aggregates node features intelligently, allowing the
    Critic to distinguish between different geometric configurations
    (e.g., clustered vs. scattered cities) that mean pooling would miss.
    """

    def __init__(
        self,
        embed_dim: int,
        c: int = 13,
        num_hidden_layers: int = 2,
        device: str = "cpu",
    ) -> None:
        super().__init__()

        # 1. Node Encoder (Replaces previous q_func)
        # We process each node independently to create a rich feature vector (embedding).
        # We explicitly define layers here to ensure output is 'embed_dim', not scalar.
        layers = []
        input_dim = c

        # Build the MLP layers
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(input_dim, embed_dim))
            layers.append(nn.ReLU())
            input_dim = embed_dim  # Next layer takes embed_dim

        # Final encoder projection (keeps dimension as embed_dim)
        layers.append(nn.Linear(embed_dim, embed_dim))

        self.node_encoder = nn.Sequential(*layers).to(device)

        # 2. Attention Pooling
        # This replaces the simple .mean()
        self.attention_pool = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=8,  # 8 heads provides stable gradients
            batch_first=True,
        )

        # 3. Glimpse Query
        # A learnable vector that asks: "What is the total value of this graph?"
        # Shape: [1, 1, embed_dim]
        self.glimpse_query = nn.Parameter(torch.randn(1, 1, embed_dim, device=device))

        # 4. Final Value Head
        # Projects the pooled graph embedding to a single scalar Value
        self.value_head = nn.Linear(embed_dim, 1).to(device)

        # Apply initialization
        self.apply(self.init_weights)

        # Specific Orthogonal Init for the final head (Critical for PPO)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)
        if self.value_head.bias is not None:
            nn.init.constant_(self.value_head.bias, 0.0)

    @staticmethod
    def init_weights(m: nn.Module) -> None:
        """Initialize weights using Kaiming uniform initialization."""
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight)
            if m.bias is not None:
                m.bias.data.fill_(0.0)  # Generally 0.0 is safer than 0.01 for deep nets

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass computing state values.
        Args:
            state: [Batch, Nodes, Features]
        Returns:
            value: [Batch] (Scalar value estimate for each problem)
        """
        # 1. Prepare Input
        # state structure assumption: [Batch, Nodes, Features]
        # We remove the first feature index as per your previous logic
        x = state[:, :, 1:]

        # 2. Encode Nodes
        # Output: [Batch, Nodes, Embed_Dim]
        node_embeddings = self.node_encoder(x)

        # 3. Attention Pooling
        # Expand the learnable query to match the batch size
        # Query: [Batch, 1, Embed_Dim]
        batch_size = x.size(0)
        query = self.glimpse_query.expand(batch_size, -1, -1)

        # Attention mechanism
        # Output graph_embedding: [Batch, 1, Embed_Dim]
        # We ignore attention weights (the second return value)
        graph_embedding, _ = self.attention_pool(
            query, node_embeddings, node_embeddings
        )

        # 4. Final Value Projection
        # Squeeze removes the sequence dimension (1) -> [Batch, Embed_Dim]
        value = self.value_head(graph_embedding.squeeze(1))

        # Remove the last dimension to return [Batch]
        return value.squeeze(-1)
