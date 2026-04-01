import math
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
        self, state: torch.Tensor, action: torch.Tensor, **kwargs
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


class CVRPActor(SAModel):
    def __init__(
        self,
        embed_dim: int = 32,
        c: int = 13,
        num_hidden_layers: int = 2,
        device: str = "cpu",
        method: str = "free",
    ) -> None:
        super().__init__(device)
        self.c1_state_dim = c
        self.method = method
        self.c2_state_dim = c * 2 - 2

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
        if device == "mps":
            pass
        else:
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
            logits[~mask] = -float("inf")
        c1, _ = self.sample_from_logits(logits, one_hot=False)

        logits = torch.ones(n_problems, x.shape[1]).to(self.generator.device)
        mask = torch.ones_like(logits, dtype=torch.bool)
        if self.method == "valid":
            mask = problem.get_action_mask(x.unsqueeze(-1), c1)
            logits[~mask] = -float("inf")  # Mask invalid actions
        else:
            arange = torch.arange(n_problems).to(logits.device)
            logits[arange, c1] = -float("inf")
            # Mask first and last logits
            if self.method == "free":
                logits[:, 0] = -float("inf")
                logits[:, -1] = -float("inf")
        c2, _ = self.sample_from_logits(logits, one_hot=False)
        action = torch.cat([c1.view(-1, 1).long(), c2.view(-1, 1).long()], dim=-1)
        return action, None, mask

    def sample(
        self, state: torch.Tensor, greedy: bool = False, problem=CVRP, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample an action pair from the current state."""
        c1_state, n_problems, x = self._prepare_features_city1(state)

        logits = self.city1_net(c1_state)[..., 0]
        mask = torch.ones_like(logits, dtype=torch.bool)
        if self.method != "rm_depot":
            tmp_mask = (x != 0).squeeze(-1)
            logits[~tmp_mask] = -float("inf")
            mask = tmp_mask

            # Mask first and last logits
            logits[:, 0] = -float("inf")
            logits[:, -1] = -float("inf")

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
            if self.method == "free":
                # Mask first and last logits
                logits[:, 0] = -float("inf")
                logits[:, -1] = -float("inf")

        c2, log_probs_c2 = self.sample_from_logits(logits, greedy=greedy, one_hot=False)

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
            # Mask first and last logits
            logits_c1[:, 0] = -float("inf")
            logits_c1[:, -1] = -float("inf")
            valid_mask_c1[:, 0] = False
            valid_mask_c1[:, -1] = False

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
            arange = torch.arange(n_problems).to(logits_c2.device)
            logits_c2[arange, taken_c1] = -float("inf")
            valid_mask_c2[arange, taken_c1] = False
            # Mask first and last logits
            if self.method == "free":
                logits_c2[:, 0] = -float("inf")
                logits_c2[:, -1] = -float("inf")
                valid_mask_c2[:, 0] = False
                valid_mask_c2[:, -1] = False

        # 3. Probabilities
        probs_c2 = torch.softmax(logits_c2, dim=-1)
        log_probs_all_c2 = torch.log_softmax(logits_c2, dim=-1)

        # 4. Entropy Calculation
        p_log_p_c2 = torch.zeros_like(probs_c2)
        p_log_p_c2[valid_mask_c2] = (
            probs_c2[valid_mask_c2] * log_probs_all_c2[valid_mask_c2]
        )
        # Safety check
        p_log_p_c2 = torch.nan_to_num(p_log_p_c2, nan=0.0)

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
        if device == "mps":
            pass
        else:
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


class PositionalEncoding(nn.Module):
    """
    Standard Sinusoidal Positional Encoding.
    Injects information about the relative or absolute position of the nodes in the sequence.
    """

    def __init__(self, embed_dim: int, max_len: int = 5000):
        super().__init__()

        # Create a matrix to hold the positional encodings
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # Calculate the frequencies for the sine and cosine waves
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim)
        )

        # Apply sine to even indices, cosine to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Shape becomes [1, max_len, embed_dim] to easily broadcast across the batch
        pe = pe.unsqueeze(0)

        # Register as a buffer so it's saved with the model but NOT updated by the optimizer
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape [Batch, Seq_Len, Embed_Dim]
        """
        # Add the positional encoding up to the current sequence length
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return x


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
        num_heads: int = 4,
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

        # Initializes the mathematical positional embeddings
        self.pos_encoder = PositionalEncoding(embed_dim=embed_dim).to(device)

        # 2. Attention Pooling (pre-norm residual)
        self.attention_pool = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(embed_dim).to(device)
        self.kv_norm = nn.LayerNorm(embed_dim).to(device)

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
        self.apply(self.init_weights)
        if device != "mps":
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
        batch_size = x.size(0)

        # 2. Encode Nodes
        # Output: [Batch, Nodes, Embed_Dim]
        node_embeddings = self.node_encoder(x)

        node_embeddings = self.pos_encoder(node_embeddings)

        # Attention Pooling (pre-norm residual)
        query = self.glimpse_query.expand(batch_size, -1, -1)
        attn_out, _ = self.attention_pool(
            self.attn_norm(query),
            self.kv_norm(node_embeddings),
            self.kv_norm(node_embeddings),
        )
        graph_embedding = query + attn_out

        # Final Value Projection
        value = self.value_head(graph_embedding.squeeze(1))
        return value.squeeze(-1)


class CVRPActorAttention(SAModel):
    """
    Attention-based actor for CVRP.

    Pipeline per forward call:
      raw features [N, D, c]
        -> node_encoder MLP  (ATTN_DIM, NUM_H_LAYERS)            -> [N, D, attn_dim]
        -> PositionalEncoding                                     -> [N, D, attn_dim]
        -> MultiheadAttention x num_attn_layers (ATTN_NUM_LAYERS) -> [N, D, attn_dim]
        -> concat(ctx[i], raw[i])                                -> [N, D, attn_dim + c]
        -> city1_scorer MLP  (EMBEDDING_DIM, NUM_H_LAYERS)       -> [N, D] logits -> c1
        -> concat(ctx[c1], ctx[i])                               -> [N, D, 2*attn_dim]
        -> city2_scorer MLP  (EMBEDDING_DIM, NUM_H_LAYERS)       -> [N, D] logits -> c2
    """

    def __init__(
        self,
        attn_dim: int = 64,
        embed_dim: int = 32,
        c: int = 13,
        num_hidden_layers: int = 1,
        num_heads: int = 4,
        num_attn_layers: int = 1,
        device: str = "cpu",
        method: str = "free",
    ) -> None:
        super().__init__(device)
        self.method = method
        self.attn_dim = attn_dim
        self.c = c

        # Node encoder: c -> attn_dim (all hidden layers use attn_dim)
        enc_layers = [nn.Linear(c, attn_dim, bias=True, device=device), nn.LeakyReLU()]
        # for _ in range(num_hidden_layers):
        #     enc_layers += [
        #         nn.Linear(attn_dim, attn_dim, bias=True, device=device),
        #         nn.LeakyReLU(),
        #     ]
        self.node_encoder = nn.Sequential(*enc_layers)

        self.pos_encoder = PositionalEncoding(embed_dim=attn_dim).to(device)

        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=attn_dim,
                num_heads=num_heads,
                batch_first=True,
                device=device,
            )
            for _ in range(num_attn_layers)
        ])

        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(attn_dim).to(device)
            for _ in range(num_attn_layers)
        ])

        # city1_scorer: concat(ctx[i], raw[i]) = attn_dim + c -> scalar
        self.city1_scorer = self._build_scorer(
            attn_dim + c, embed_dim, num_hidden_layers, device
        )

        # city2_scorer: concat(c1_ctx, c1_raw, candidate_ctx, candidate_raw) = 2*attn_dim + 2*c -> scalar
        self.city2_scorer = self._build_scorer(
            2 * attn_dim + 2 * c, 2*embed_dim, num_hidden_layers, device
        )

        if device != "mps":
            self.node_encoder.apply(self.init_weights)
            self.city1_scorer.apply(self.init_weights)
            nn.init.orthogonal_(self.city1_scorer[-1].weight, gain=0.01)
            self.city2_scorer.apply(self.init_weights)
            nn.init.orthogonal_(self.city2_scorer[-1].weight, gain=0.01)

    @staticmethod
    def _build_scorer(
        input_dim: int, embed_dim: int, num_hidden_layers: int, device: str
    ) -> nn.Sequential:
        """MLP: input_dim -> embed_dim (x num_hidden_layers) -> scalar."""
        layers = [nn.Linear(input_dim, embed_dim, bias=True, device=device), nn.LeakyReLU()]
        for _ in range(num_hidden_layers):
            layers += [
                nn.Linear(embed_dim, embed_dim, bias=True, device=device),
                nn.LeakyReLU(),
            ]
        layers.append(nn.Linear(embed_dim, 1, bias=False, device=device))
        return nn.Sequential(*layers)

    def sample_from_logits(
        self, logits: torch.Tensor, greedy: bool = False, one_hot: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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

    def _encode(
        self, state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Shared encoder: runs node encoder + positional encoding + all attention layers.

        Returns:
            ctx:      [N, D, attn_dim]  per-node context vectors
            features: [N, D, c]         raw node features (state[:, :, 1:])
            x:        [N, D, 1]         route ids       (state[:, :, :1])
        """
        x = state[:, :, :1]         # [N, D, 1]
        features = state[:, :, 1:]  # [N, D, c]
        node_emb = self.node_encoder(features)    # [N, D, attn_dim]
        node_emb = self.pos_encoder(node_emb)     # [N, D, attn_dim]
        ctx = node_emb
        for attn, norm in zip(self.attention_layers, self.layer_norms):
            ctx_normed = norm(ctx)
            attn_out, _ = attn(ctx_normed, ctx_normed, ctx_normed)
            ctx = ctx + attn_out                  # [N, D, attn_dim]
        return ctx, features, x

    def _logits_c1(self, ctx: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        inp = torch.cat([ctx, features], dim=-1)  # [N, D, attn_dim + c]
        return self.city1_scorer(inp)[..., 0]     # [N, D]

    def _logits_c2(
        self, ctx: torch.Tensor, c1: torch.Tensor, features: torch.Tensor
    ) -> torch.Tensor:
        arange = torch.arange(ctx.shape[0], device=ctx.device)
        c1_ctx = ctx[arange, c1]                                              # [N, attn_dim]
        c1_raw = features[arange, c1]                                         # [N, c]
        c1_ctx_exp = c1_ctx[:, None, :].expand_as(ctx)                        # [N, D, attn_dim]
        c1_raw_exp = c1_raw[:, None, :].expand(
            ctx.shape[0], ctx.shape[1], features.shape[-1]
        )                                                                     # [N, D, c]
        inp = torch.cat([c1_ctx_exp, c1_raw_exp, ctx, features], dim=-1)     # [N, D, 2*attn_dim + 2*c]
        return self.city2_scorer(inp)[..., 0]                                 # [N, D]

    def sample(
        self, state: torch.Tensor, greedy: bool = False, problem=CVRP, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ctx, features, x = self._encode(state)
        n_problems = state.shape[0]

        # City 1
        logits = self._logits_c1(ctx, features)
        mask = torch.ones_like(logits, dtype=torch.bool)
        if self.method != "rm_depot":
            tmp_mask = (x != 0).squeeze(-1)
            logits[~tmp_mask] = -float("inf")
            logits[:, 0] = -float("inf")
            logits[:, -1] = -float("inf")
            mask = tmp_mask

        c1, log_probs_c1 = self.sample_from_logits(logits, greedy=greedy)

        # City 2
        logits = self._logits_c2(ctx, c1, features)
        mask = torch.ones_like(logits, dtype=torch.bool)
        if self.method == "valid":
            mask = problem.get_action_mask(solution=x, node_pos=c1)
            logits[~mask] = -float("inf")
        else:
            arange = torch.arange(n_problems, device=logits.device)
            logits[arange, c1] = -float("inf")
            if self.method == "free":
                logits[:, 0] = -float("inf")
                logits[:, -1] = -float("inf")

        c2, log_probs_c2 = self.sample_from_logits(logits, greedy=greedy)

        action = torch.cat([c1.view(-1, 1).long(), c2.view(-1, 1).long()], dim=-1)
        return action, (log_probs_c1 + log_probs_c2)[..., 0], mask

    def evaluate(
        self, state: torch.Tensor, action: torch.Tensor, mask: torch.Tensor, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        ctx, features, x = self._encode(state)
        n_problems = state.shape[0]
        taken_c1 = action[:, 0]
        taken_c2 = action[:, 1]

        # City 1
        logits_c1 = self._logits_c1(ctx, features)
        valid_mask_c1 = torch.ones_like(logits_c1, dtype=torch.bool)
        if self.method != "rm_depot":
            tmp_mask = (x != 0).squeeze(-1)
            logits_c1[~tmp_mask] = -float("inf")
            logits_c1[:, 0] = -float("inf")
            logits_c1[:, -1] = -float("inf")
            valid_mask_c1 = tmp_mask
            valid_mask_c1[:, 0] = False
            valid_mask_c1[:, -1] = False

        probs_c1 = torch.softmax(logits_c1, dim=-1)
        log_probs_all_c1 = torch.log_softmax(logits_c1, dim=-1)
        p_log_p_c1 = torch.zeros_like(probs_c1)
        p_log_p_c1[valid_mask_c1] = probs_c1[valid_mask_c1] * log_probs_all_c1[valid_mask_c1]
        entropy_c1 = -p_log_p_c1.sum(dim=-1)
        chosen_log_prob_c1 = log_probs_all_c1.gather(1, taken_c1.view(-1, 1)).squeeze(-1)

        # City 2
        logits_c2 = self._logits_c2(ctx, taken_c1, features)
        valid_mask_c2 = torch.ones_like(logits_c2, dtype=torch.bool)
        if self.method == "valid":
            logits_c2[~mask] = -float("inf")
            valid_mask_c2 = mask
        else:
            arange = torch.arange(n_problems, device=logits_c2.device)
            logits_c2[arange, taken_c1] = -float("inf")
            valid_mask_c2[arange, taken_c1] = False
            if self.method == "free":
                logits_c2[:, 0] = -float("inf")
                logits_c2[:, -1] = -float("inf")
                valid_mask_c2[:, 0] = False
                valid_mask_c2[:, -1] = False

        probs_c2 = torch.softmax(logits_c2, dim=-1)
        log_probs_all_c2 = torch.log_softmax(logits_c2, dim=-1)
        p_log_p_c2 = torch.zeros_like(probs_c2)
        p_log_p_c2[valid_mask_c2] = probs_c2[valid_mask_c2] * log_probs_all_c2[valid_mask_c2]
        p_log_p_c2 = torch.nan_to_num(p_log_p_c2, nan=0.0)
        entropy_c2 = -p_log_p_c2.sum(dim=-1)
        chosen_log_prob_c2 = log_probs_all_c2.gather(1, taken_c2.view(-1, 1)).squeeze(-1)

        return chosen_log_prob_c1 + chosen_log_prob_c2, entropy_c1 + entropy_c2

    def get_logits(
        self, state: torch.Tensor, action: torch.Tensor, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        ctx, features, _ = self._encode(state)
        c1 = action[:, 0]
        log_probs_c1 = torch.log(torch.softmax(self._logits_c1(ctx, features), dim=-1))
        log_probs_c2 = torch.log(torch.softmax(self._logits_c2(ctx, c1, features), dim=-1))
        return log_probs_c1, log_probs_c2

    def baseline_sample(
        self, state: torch.Tensor, problem=CVRP, **kwargs
    ) -> Tuple[torch.Tensor, None, torch.Tensor]:
        n_problems, problem_dim, _ = state.shape
        x = state[:, :, 0]
        mask = x != 0

        logits = torch.ones(n_problems, problem_dim, device=self.generator.device)
        if self.method != "rm_depot":
            logits[~mask] = -float("inf")

        c1, _ = self.sample_from_logits(logits)

        logits = torch.ones(n_problems, problem_dim, device=self.generator.device)
        mask = torch.ones_like(logits, dtype=torch.bool)
        if self.method == "valid":
            mask = problem.get_action_mask(x.unsqueeze(-1), c1)
            logits[~mask] = -float("inf")
        else:
            arange = torch.arange(n_problems, device=logits.device)
            logits[arange, c1] = -float("inf")
            if self.method == "free":
                logits[:, 0] = -float("inf")
                logits[:, -1] = -float("inf")

        c2, _ = self.sample_from_logits(logits)
        action = torch.cat([c1.view(-1, 1).long(), c2.view(-1, 1).long()], dim=-1)
        return action, None, mask
