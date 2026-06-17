# src/model/actors.py
from typing import Any

import torch
import torch.nn as nn

from problem import CVRP
from utils import repeat_to

from .base import PositionalEncoding, SAModel, build_mlp


class CVRPActor(SAModel):
    """
    Standard MLP-based Actor model for the Capacitated Vehicle Routing Problem (CVRP).
    Selects two cities (nodes) to perform a local search operation on.
    """

    def __init__(
        self,
        embed_dim: int = 32,
        c: int = 13,
        num_hidden_layers: int = 2,
        device: str = "cpu",
        method: str = "free",
        cond_rank: bool = False,
        cond_detour: bool = False,
    ) -> None:
        """
        Initialize the CVRPActor.

        Args:
            embed_dim (int): Dimension of hidden layers.
            c (int): Input feature dimension per node.
            num_hidden_layers (int): Number of hidden layers in the MLPs.
            device (str): Device to run the computations on.
            method (str): Masking method to apply ('free', 'valid', 'rm_depot').
            cond_rank (bool): If True, condition city-2 selection on the bidirectional
                distance-rank of the two edges an insertion creates (4 extra channels:
                rank(u, v), rank(v, u), rank(u, succ(v)), rank(succ(v), u)).
            cond_detour (bool): If True, condition city-2 selection on the insertion
                cost of placing city 1 after the candidate (2 extra channels:
                insert_cost, net_delta). Insertion-specific (see
                CVRP.conditional_insertion_detour).
        """
        super().__init__(device)
        self.c1_state_dim = c
        self.method = method
        self.cond_rank = cond_rank
        self.cond_detour = cond_detour
        # Features for city 2 include city 1's features -2 to not include meta features twice, resulting in c + c - 2.
        # +4 when cond_rank is on (edge ranks), +2 when cond_detour is on (insertion cost).
        self.c2_state_dim = c * 2 - 2 + (4 if cond_rank else 0) + (2 if cond_detour else 0)

        self.city1_net = build_mlp(self.c1_state_dim, embed_dim, num_hidden_layers, device)
        self.city2_net = build_mlp(self.c2_state_dim, embed_dim, num_hidden_layers, device)

        if device != "mps":
            self.city1_net.apply(self.init_weights)
            last_layer_c1 = self.city1_net[-1]
            if isinstance(last_layer_c1, nn.Linear):
                nn.init.orthogonal_(last_layer_c1.weight, gain=0.01)
                if last_layer_c1.bias is not None:
                    nn.init.constant_(last_layer_c1.bias, 0.0)

            self.city2_net.apply(self.init_weights)
            last_layer_c2 = self.city2_net[-1]
            if isinstance(last_layer_c2, nn.Linear):
                nn.init.orthogonal_(last_layer_c2.weight, gain=0.01)
                if last_layer_c2.bias is not None:
                    nn.init.constant_(last_layer_c2.bias, 0.0)

    def get_logits(
        self, state: torch.Tensor, action: torch.Tensor, **kwargs: Any
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute logits and log probabilities for given state and action.

        Args:
            state (torch.Tensor): Current state, shape: (batch_size, problem_dim, features)
            action (torch.Tensor): Given actions, shape: (batch_size, 2)

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: log_probs_c1 and log_probs_c2
                                               Shapes: both are (batch_size, problem_dim)
        """
        c1_state, n_problems, routes_ids = self._prepare_features_city1(state)
        # c1_state shape: (batch_size, problem_dim, c) or condensed
        c1 = action[:, 0]  # shape: (batch_size,)

        logits_c1 = self.city1_net(c1_state)[..., 0]  # shape: (batch_size, problem_dim)
        probs_c1 = torch.softmax(logits_c1, dim=-1)  # shape: (batch_size, problem_dim)
        log_probs_c1 = torch.log(probs_c1)  # shape: (batch_size, problem_dim)

        cond_rank = self._conditional_features(kwargs["problem"], routes_ids, c1)
        c2_state = self._prepare_features_city2(c1_state, c1, n_problems, cond_rank)
        # c2_state shape: (batch_size, problem_dim, c2_state_dim)

        logits_c2 = self.city2_net(c2_state)[..., 0]  # shape: (batch_size, problem_dim)
        probs_c2 = torch.softmax(logits_c2, dim=-1)  # shape: (batch_size, problem_dim)
        log_probs_c2 = torch.log(probs_c2)  # shape: (batch_size, problem_dim)

        return log_probs_c1, log_probs_c2

    def baseline_sample(
        self, state: torch.Tensor, problem: Any = CVRP, **kwargs: Any
    ) -> tuple[torch.Tensor, None, torch.Tensor]:
        """
        Generate a baseline sample using uniform probabilities.

        Args:
            state (torch.Tensor): Current state, shape: (batch_size, problem_dim, features)
            problem: The problem class/instance for mask generation.

        Returns:
            Tuple[torch.Tensor, None, torch.Tensor]: actions, None (no log_probs), mask
        """
        n_problems, problem_dim, _ = state.shape
        x = state[:, :, 0]  # shape: (batch_size, problem_dim)

        # Identify non-zero routes. mask shape: (batch_size, problem_dim)
        mask = x.squeeze(-1) != 0
        if self.method == "rm_depot":
            x_filtered = x[mask].view(n_problems, -1)
            logits_c1 = torch.ones(n_problems, x_filtered.shape[1], device=self.generator.device)
        else:
            logits_c1 = torch.ones(n_problems, x.shape[1], device=self.generator.device)
            logits_c1[~mask] = -float("inf")

        c1, _ = self.sample_from_logits(logits_c1, one_hot=False)  # shape: (batch_size,)

        logits_c2 = torch.ones(n_problems, x.shape[1], device=self.generator.device)
        mask_c2 = torch.ones_like(logits_c2, dtype=torch.bool)
        if self.method == "valid":
            mask_c2 = problem.get_action_mask(
                x.unsqueeze(-1), c1
            )  # shape: (batch_size, problem_dim)
            logits_c2[~mask_c2] = -float("inf")
        else:
            arange = torch.arange(n_problems, device=logits_c2.device)
            logits_c2[arange, c1] = -float("inf")
            if self.method == "free":
                logits_c2[:, 0] = -float("inf")
                logits_c2[:, -1] = -float("inf")

        c2, _ = self.sample_from_logits(logits_c2, one_hot=False)  # shape: (batch_size,)

        # action shape: (batch_size, 2)
        action = torch.cat([c1.view(-1, 1).long(), c2.view(-1, 1).long()], dim=-1)
        cond_rank = self._conditional_features(problem, x, c1)
        return action, None, mask_c2, cond_rank

    def sample(
        self, state: torch.Tensor, greedy: bool = False, problem: Any = CVRP, **kwargs: Any
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample an action pair from the current state.

        Args:
            state (torch.Tensor): State tensor, shape: (batch_size, problem_dim, features)
            greedy (bool): If True, pick max logit instead of sampling.
            problem: Problem instance for obtaining action mask.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                actions (batch_size, 2), chosen_log_probs (batch_size,), mask (batch_size, problem_dim)
        """
        c1_state, n_problems, x = self._prepare_features_city1(state)
        # c1_state shape: (batch_size, problem_dim, c)

        logits_c1 = self.city1_net(c1_state)[..., 0]  # shape: (batch_size, problem_dim)
        logits_c1, _ = self._apply_mask_c1(logits_c1, x, self.method)
        c1, log_probs_c1 = self.sample_from_logits(logits_c1, greedy=greedy, one_hot=False)

        cond_rank = self._conditional_features(problem, x, c1)
        c2_state = self._prepare_features_city2(c1_state, c1, n_problems, cond_rank)
        # c2_state shape: (batch_size, problem_dim, c2_state_dim)

        logits_c2 = self.city2_net(c2_state)[..., 0]  # shape: (batch_size, problem_dim)
        ext_mask = (
            problem.get_action_mask(solution=x, node_pos=c1) if self.method == "valid" else None
        )
        logits_c2, mask = self._apply_mask_c2(
            logits_c2, c1, self.method, n_problems, external_mask=ext_mask
        )
        c2, log_probs_c2 = self.sample_from_logits(logits_c2, greedy=greedy, one_hot=False)

        # concatenate sampled cities
        action = torch.cat(
            [c1.view(-1, 1).long(), c2.view(-1, 1).long()], dim=-1
        )  # shape: (batch_size, 2)
        log_probs = log_probs_c1 + log_probs_c2  # shape: (batch_size, 1) or (batch_size,)
        return action, log_probs[..., 0] if log_probs.dim() > 1 else log_probs, mask, cond_rank

    def evaluate(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        mask: torch.Tensor,
        cond_rank: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate actions to get their Log-Probabilities and the Entropy of the distribution.

        Args:
            state (torch.Tensor): Shape (batch_size, problem_dim, features)
            action (torch.Tensor): Shape (batch_size, 2) containing [City1_Index, City2_Index]
            mask (torch.Tensor): Tensor indicating valid choices for the second city
            cond_rank (torch.Tensor | None): Stored bidirectional distance-rank of the two
                insertion edges (u, v) and (u, succ(v)), shape (batch_size, problem_dim, 4);
                None when disabled.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                log_probs: Shape (batch_size,) - The log-likelihood of the specific actions taken.
                total_entropy: Shape (batch_size,) - The entropy of the entire policy distribution.
        """
        c1_state, n_problems, x = self._prepare_features_city1(state)
        taken_c1 = action[:, 0]
        taken_c2 = action[:, 1]

        # Evaluate City 1
        logits_c1 = self.city1_net(c1_state)[..., 0]
        logits_c1, valid_mask_c1 = self._apply_mask_c1(logits_c1, x, self.method)

        probs_c1 = torch.softmax(logits_c1, dim=-1)  # shape: (batch_size, problem_dim)
        log_probs_all_c1 = torch.log_softmax(logits_c1, dim=-1)  # shape: (batch_size, problem_dim)
        p_log_p_c1 = torch.zeros_like(probs_c1)
        p_log_p_c1[valid_mask_c1] = probs_c1[valid_mask_c1] * log_probs_all_c1[valid_mask_c1]
        entropy_c1 = -p_log_p_c1.sum(dim=-1)  # shape: (batch_size,)
        chosen_log_prob_c1 = log_probs_all_c1.gather(1, taken_c1.view(-1, 1)).squeeze(-1)

        # Evaluate City 2
        c2_state = self._prepare_features_city2(c1_state, taken_c1, n_problems, cond_rank)
        logits_c2 = self.city2_net(c2_state)[..., 0]
        logits_c2, valid_mask_c2 = self._apply_mask_c2(
            logits_c2,
            taken_c1,
            self.method,
            n_problems,
            external_mask=mask if self.method == "valid" else None,
        )

        probs_c2 = torch.softmax(logits_c2, dim=-1)  # shape: (batch_size, problem_dim)
        log_probs_all_c2 = torch.log_softmax(logits_c2, dim=-1)  # shape: (batch_size, problem_dim)
        p_log_p_c2 = torch.zeros_like(probs_c2)
        p_log_p_c2[valid_mask_c2] = probs_c2[valid_mask_c2] * log_probs_all_c2[valid_mask_c2]
        p_log_p_c2 = torch.nan_to_num(p_log_p_c2, nan=0.0)
        entropy_c2 = -p_log_p_c2.sum(dim=-1)  # shape: (batch_size,)
        chosen_log_prob_c2 = log_probs_all_c2.gather(1, taken_c2.view(-1, 1)).squeeze(-1)

        # Ensure shapes are correctly aligned
        return chosen_log_prob_c1 + chosen_log_prob_c2, entropy_c1 + entropy_c2

    def _prepare_features_city1(
        self, state: torch.Tensor
    ) -> tuple[torch.Tensor, int, torch.Tensor]:
        """
        Prepare initial state features for processing the first city.

        Args:
            state (torch.Tensor): Shape (batch_size, problem_dim, features)

        Returns:
            Tuple[torch.Tensor, int, torch.Tensor]:
                c_state: Features per node, shape (batch_size, problem_dim, c)
                n_problems: Batch size
                x: Extracted route ids, shape (batch_size, problem_dim, 1)
        """
        n_problems, problem_dim, dim = state.shape
        x = state[:, :, :1]  # shape: (batch_size, problem_dim, 1)
        c_state = state[:, :, 1:]  # shape: (batch_size, problem_dim, features - 1)

        if self.method == "rm_depot":
            mask = x.squeeze(-1) != 0
            c_state = c_state[mask].view(n_problems, -1, c_state.size(-1))

        return c_state, n_problems, x

    def _prepare_features_city2(
        self,
        c1_state: torch.Tensor,
        c1: torch.Tensor,
        n_problems: int,
        cond_rank: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Prepare features for evaluating the second city conditionned on the first city.

        Args:
            c1_state (torch.Tensor): Node features, shape (batch_size, problem_dim, c)
            c1 (torch.Tensor): Selected city 1 inputs, shape (batch_size,)
            n_problems (int): Batch size
            cond_rank (torch.Tensor | None): Optional bidirectional distance-rank of the two
                insertion edges (u, v) and (u, succ(v)), shape (batch_size, problem_dim, 4).

        Returns:
            torch.Tensor: Combined state features, shape (batch_size, problem_dim, c2_state_dim)
        """
        arange = torch.arange(n_problems)
        # Get feature vector for chosen city 1
        c1_val = c1_state[arange, c1]  # shape: (batch_size, c)
        base = torch.cat([c1_val], -1)[:, None, :]  # shape: (batch_size, 1, c)
        base = repeat_to(base, c1_state)  # shape: (batch_size, problem_dim, c)

        # Omit last two features from c1_state and combine with chosen c1 base
        c1_state_trunc = c1_state[:, :, :-2]  # shape: (batch_size, problem_dim, c-2)
        c2_state = torch.cat([base, c1_state_trunc], -1)  # shape: (batch_size, problem_dim, c*2-2)
        if cond_rank is not None:
            c2_state = torch.cat([c2_state, cond_rank], -1)  # append conditional city-2 features
        return c2_state

    def _conditional_features(
        self, problem: Any, x: torch.Tensor, c1: torch.Tensor
    ) -> torch.Tensor | None:
        """
        Build the conditional city-2 features that depend on the chosen city 1.

        Concatenates whichever conditional blocks are enabled into a single tensor
        (carried through the replay buffer / PPO under the 'cond_rank' slot):
          - cond_rank  -> 4 channels (bidirectional edge ranks)
          - cond_detour -> 2 channels (insert_cost, net_delta)

        Args:
            problem: CVRP instance exposing the conditional feature methods.
            x (torch.Tensor): Route-id sequence, shape (batch_size, problem_dim, 1).
            c1 (torch.Tensor): Selected city-1 positions, shape (batch_size,).

        Returns:
            torch.Tensor | None: (batch_size, problem_dim, n_cond_channels) or None when
                no conditional feature is enabled.
        """
        feats = []
        if self.cond_rank:
            feats.append(problem.conditional_neighbor_rank(x, c1))
        if self.cond_detour:
            feats.append(problem.conditional_insertion_detour(x, c1))
        return torch.cat(feats, dim=-1) if feats else None


class CVRPActorAttention(SAModel):
    """
    Attention-based actor for the Capacitated Vehicle Routing Problem (CVRP).

    Pipeline per forward call:
      raw features (batch_size, problem_dim, c)
        -> node_encoder MLP                              -> (batch_size, problem_dim, attn_dim)
        -> PositionalEncoding                            -> (batch_size, problem_dim, attn_dim)
        -> MultiheadAttention x num_attn_layers          -> (batch_size, problem_dim, attn_dim)
        -> concat(ctx, raw)                              -> (batch_size, problem_dim, attn_dim + c)
        -> city1_scorer MLP                              -> (batch_size, problem_dim) logits -> c1
        -> concat(c1_ctx, c1_raw, ctx, raw)              -> (batch_size, problem_dim, 2*attn_dim + 2*c)
        -> city2_scorer MLP                              -> (batch_size, problem_dim) logits -> c2
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
        """
        Initialize the CVRPActorAttention.

        Args:
            attn_dim (int): Dimension of attention embeddings.
            embed_dim (int): Dimension of final scorer MLPs.
            c (int): Input feature dimension.
            num_hidden_layers (int): Number of hidden layers in MLP scorers.
            num_heads (int): Number of attention heads.
            num_attn_layers (int): Number of attention layers.
            device (str): Device to run computations on.
            method (str): Masking method.
        """
        super().__init__(device)
        self.method = method
        self.attn_dim = attn_dim
        self.c = c

        # Node encoder: mapped from features `c` directly to `attn_dim`
        enc_layers: list[nn.Module] = [
            nn.Linear(c, attn_dim, bias=True, device=device),
            nn.LeakyReLU(),
        ]
        self.node_encoder = nn.Sequential(*enc_layers)
        self.pos_encoder = PositionalEncoding(embed_dim=attn_dim).to(device)

        self.attention_layers = nn.ModuleList(
            [
                nn.MultiheadAttention(
                    embed_dim=attn_dim,
                    num_heads=num_heads,
                    batch_first=True,
                    device=device,
                )
                for _ in range(num_attn_layers)
            ]
        )

        self.layer_norms = nn.ModuleList(
            [nn.LayerNorm(attn_dim).to(device) for _ in range(num_attn_layers)]
        )

        # Scorer mapping Context + Features -> Logits
        self.city1_scorer = build_mlp(attn_dim + c, embed_dim, num_hidden_layers, device)
        self.city2_scorer = build_mlp(
            2 * attn_dim + 2 * c, 2 * embed_dim, num_hidden_layers, device
        )

        if device != "mps":
            self.node_encoder.apply(self.init_weights)
            self.city1_scorer.apply(self.init_weights)
            if isinstance(self.city1_scorer[-1], nn.Linear):
                nn.init.orthogonal_(self.city1_scorer[-1].weight, gain=0.01)

            self.city2_scorer.apply(self.init_weights)
            if isinstance(self.city2_scorer[-1], nn.Linear):
                nn.init.orthogonal_(self.city2_scorer[-1].weight, gain=0.01)

    def _encode(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Apply node, positional encoding and self-attention to generate contexts.

        Args:
            state (torch.Tensor): Network state inputs, shape (batch_size, problem_dim, features)

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                ctx: Context embeddings, shape (batch_size, problem_dim, attn_dim)
                features: Raw node features, shape (batch_size, problem_dim, c)
                x: Route IDs, shape (batch_size, problem_dim, 1)
        """
        x = state[:, :, :1]  # shape: (batch_size, problem_dim, 1)
        features = state[:, :, 1:]  # shape: (batch_size, problem_dim, c)

        node_emb = self.node_encoder(features)  # shape: (batch_size, problem_dim, attn_dim)
        node_emb = self.pos_encoder(node_emb)  # shape: (batch_size, problem_dim, attn_dim)

        ctx = node_emb
        for attn, norm in zip(self.attention_layers, self.layer_norms):
            ctx_normed = norm(ctx)
            # Self-attention over node context
            attn_out, _ = attn(
                ctx_normed, ctx_normed, ctx_normed
            )  # shape: (batch_size, problem_dim, attn_dim)
            ctx = ctx + attn_out

        return ctx, features, x

    def _logits_c1(self, ctx: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        """Helper to get city 1 logits."""
        inp = torch.cat([ctx, features], dim=-1)  # shape: (batch_size, problem_dim, attn_dim + c)
        return self.city1_scorer(inp)[..., 0]  # shape: (batch_size, problem_dim)

    def _logits_c2(
        self, ctx: torch.Tensor, c1: torch.Tensor, features: torch.Tensor
    ) -> torch.Tensor:
        """Helper to condition and evaluate city 2 logits."""
        batch_size, problem_dim, _ = ctx.shape
        arange = torch.arange(batch_size, device=ctx.device)

        c1_ctx = ctx[arange, c1]  # shape: (batch_size, attn_dim)
        c1_raw = features[arange, c1]  # shape: (batch_size, c)

        c1_ctx_exp = c1_ctx[:, None, :].expand_as(ctx)  # shape: (batch_size, problem_dim, attn_dim)
        c1_raw_exp = c1_raw[:, None, :].expand(
            batch_size, problem_dim, features.shape[-1]
        )  # shape: (batch_size, problem_dim, c)

        inp = torch.cat(
            [c1_ctx_exp, c1_raw_exp, ctx, features], dim=-1
        )  # shape: (batch_size, problem_dim, 2*attn_dim + 2*c)
        return self.city2_scorer(inp)[..., 0]  # shape: (batch_size, problem_dim)

    def get_logits(
        self, state: torch.Tensor, action: torch.Tensor, **kwargs: Any
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ctx, features, _ = self._encode(state)
        c1 = action[:, 0]
        log_probs_c1 = torch.log(torch.softmax(self._logits_c1(ctx, features), dim=-1))
        log_probs_c2 = torch.log(torch.softmax(self._logits_c2(ctx, c1, features), dim=-1))
        return log_probs_c1, log_probs_c2

    def baseline_sample(
        self, state: torch.Tensor, problem: Any = CVRP, **kwargs: Any
    ) -> tuple[torch.Tensor, None, torch.Tensor]:
        n_problems, problem_dim, _ = state.shape
        x = state[:, :, 0]
        mask = x != 0

        logits_c1 = torch.ones(n_problems, problem_dim, device=self.generator.device)
        if self.method != "rm_depot":
            logits_c1[~mask] = -float("inf")
        c1, _ = self.sample_from_logits(logits_c1)

        logits_c2 = torch.ones(n_problems, problem_dim, device=self.generator.device)
        mask_c2 = torch.ones_like(logits_c2, dtype=torch.bool)
        if self.method == "valid":
            mask_c2 = problem.get_action_mask(x.unsqueeze(-1), c1)
            logits_c2[~mask_c2] = -float("inf")
        else:
            arange = torch.arange(n_problems, device=logits_c2.device)
            logits_c2[arange, c1] = -float("inf")
            if self.method == "free":
                logits_c2[:, 0] = -float("inf")
                logits_c2[:, -1] = -float("inf")

        c2, _ = self.sample_from_logits(logits_c2)
        action = torch.cat([c1.view(-1, 1).long(), c2.view(-1, 1).long()], dim=-1)
        return action, None, mask_c2

    def sample(
        self, state: torch.Tensor, greedy: bool = False, problem: Any = CVRP, **kwargs: Any
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ctx, features, x = self._encode(state)
        n_problems = state.shape[0]

        logits_c1 = self._logits_c1(ctx, features)
        logits_c1, _ = self._apply_mask_c1(logits_c1, x, self.method)
        c1, log_probs_c1 = self.sample_from_logits(logits_c1, greedy=greedy)

        logits_c2 = self._logits_c2(ctx, c1, features)
        ext_mask = (
            problem.get_action_mask(solution=x, node_pos=c1) if self.method == "valid" else None
        )
        logits_c2, mask = self._apply_mask_c2(
            logits_c2, c1, self.method, n_problems, external_mask=ext_mask
        )
        c2, log_probs_c2 = self.sample_from_logits(logits_c2, greedy=greedy)

        action = torch.cat([c1.view(-1, 1).long(), c2.view(-1, 1).long()], dim=-1)
        log_probs_sum = log_probs_c1 + log_probs_c2
        return action, log_probs_sum[..., 0] if log_probs_sum.dim() > 1 else log_probs_sum, mask

    def evaluate(
        self, state: torch.Tensor, action: torch.Tensor, mask: torch.Tensor, **kwargs: Any
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ctx, features, x = self._encode(state)
        n_problems = state.shape[0]
        taken_c1 = action[:, 0]
        taken_c2 = action[:, 1]

        # City 1
        logits_c1 = self._logits_c1(ctx, features)
        logits_c1, valid_mask_c1 = self._apply_mask_c1(logits_c1, x, self.method)

        probs_c1 = torch.softmax(logits_c1, dim=-1)
        log_probs_all_c1 = torch.log_softmax(logits_c1, dim=-1)
        p_log_p_c1 = torch.zeros_like(probs_c1)
        p_log_p_c1[valid_mask_c1] = probs_c1[valid_mask_c1] * log_probs_all_c1[valid_mask_c1]
        entropy_c1 = -p_log_p_c1.sum(dim=-1)
        chosen_log_prob_c1 = log_probs_all_c1.gather(1, taken_c1.view(-1, 1)).squeeze(-1)

        # City 2
        logits_c2 = self._logits_c2(ctx, taken_c1, features)
        logits_c2, valid_mask_c2 = self._apply_mask_c2(
            logits_c2,
            taken_c1,
            self.method,
            n_problems,
            external_mask=mask if self.method == "valid" else None,
        )

        probs_c2 = torch.softmax(logits_c2, dim=-1)
        log_probs_all_c2 = torch.log_softmax(logits_c2, dim=-1)
        p_log_p_c2 = torch.zeros_like(probs_c2)
        p_log_p_c2[valid_mask_c2] = probs_c2[valid_mask_c2] * log_probs_all_c2[valid_mask_c2]
        p_log_p_c2 = torch.nan_to_num(p_log_p_c2, nan=0.0)
        entropy_c2 = -p_log_p_c2.sum(dim=-1)
        chosen_log_prob_c2 = log_probs_all_c2.gather(1, taken_c2.view(-1, 1)).squeeze(-1)

        return chosen_log_prob_c1 + chosen_log_prob_c2, entropy_c1 + entropy_c2
