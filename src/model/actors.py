# src/model/actors.py
from typing import Tuple

import torch
import torch.nn as nn

from problem import CVRP
from utils import repeat_to

from .base import PositionalEncoding, SAModel, build_mlp


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

        self.city1_net = build_mlp(
            self.c1_state_dim, embed_dim, num_hidden_layers, device
        )
        self.city2_net = build_mlp(
            self.c2_state_dim, embed_dim, num_hidden_layers, device
        )

        if device != "mps":
            self.city1_net.apply(self.init_weights)
            last_layer = self.city1_net[-1]
            nn.init.orthogonal_(last_layer.weight, gain=0.01)
            if last_layer.bias is not None:
                nn.init.constant_(last_layer.bias, 0.0)

            self.city2_net.apply(self.init_weights)
            last_layer = self.city2_net[-1]
            nn.init.orthogonal_(last_layer.weight, gain=0.01)
            if last_layer.bias is not None:
                nn.init.constant_(last_layer.bias, 0.0)

    def get_logits(
        self, state: torch.Tensor, action: torch.Tensor, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute logits and log probabilities for given state and action."""
        c1_state, n_problems, routes_ids = self._prepare_features_city1(state)
        c1 = action[:, 0]
        logits = self.city1_net(c1_state)[..., 0]
        probs = torch.softmax(logits, dim=-1)
        log_probs_c1 = torch.log(probs)
        c2_state = self._prepare_features_city2(c1_state, c1, n_problems)
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
            logits[~mask] = -float("inf")
        else:
            arange = torch.arange(n_problems).to(logits.device)
            logits[arange, c1] = -float("inf")
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
        logits, _ = self._apply_mask_c1(logits, x, self.method)
        c1, log_probs_c1 = self.sample_from_logits(logits, greedy=greedy, one_hot=False)

        c2_state = self._prepare_features_city2(c1_state, c1, n_problems)
        logits = self.city2_net(c2_state)[..., 0]
        ext_mask = (
            problem.get_action_mask(solution=x, node_pos=c1)
            if self.method == "valid"
            else None
        )
        logits, mask = self._apply_mask_c2(
            logits, c1, self.method, n_problems, external_mask=ext_mask
        )
        c2, log_probs_c2 = self.sample_from_logits(logits, greedy=greedy, one_hot=False)

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
            mask: Tensor indicating valid choices for the second city (used only when
                  method='valid'; ignored otherwise)

        Returns:
            log_probs: The log-likelihood of the specific actions taken.
            total_entropy: The entropy (uncertainty) of the entire policy distribution.
        """
        c1_state, n_problems, x = self._prepare_features_city1(state)
        taken_c1 = action[:, 0]
        taken_c2 = action[:, 1]

        # City 1
        logits_c1 = self.city1_net(c1_state)[..., 0]
        logits_c1, valid_mask_c1 = self._apply_mask_c1(logits_c1, x, self.method)

        probs_c1 = torch.softmax(logits_c1, dim=-1)
        log_probs_all_c1 = torch.log_softmax(logits_c1, dim=-1)
        p_log_p_c1 = torch.zeros_like(probs_c1)
        p_log_p_c1[valid_mask_c1] = (
            probs_c1[valid_mask_c1] * log_probs_all_c1[valid_mask_c1]
        )
        entropy_c1 = -p_log_p_c1.sum(dim=-1)
        chosen_log_prob_c1 = log_probs_all_c1.gather(1, taken_c1.view(-1, 1)).squeeze(
            -1
        )

        # City 2
        c2_state = self._prepare_features_city2(c1_state, taken_c1, n_problems)
        logits_c2 = self.city2_net(c2_state)[..., 0]
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
        p_log_p_c2[valid_mask_c2] = (
            probs_c2[valid_mask_c2] * log_probs_all_c2[valid_mask_c2]
        )
        p_log_p_c2 = torch.nan_to_num(p_log_p_c2, nan=0.0)
        entropy_c2 = -p_log_p_c2.sum(dim=-1)
        chosen_log_prob_c2 = log_probs_all_c2.gather(1, taken_c2.view(-1, 1)).squeeze(
            -1
        )

        return chosen_log_prob_c1 + chosen_log_prob_c2, entropy_c1 + entropy_c2

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
        arange = torch.arange(n_problems)
        c1_val = c1_state[arange, c1]
        base = torch.cat([c1_val], -1)[:, None, :]
        base = repeat_to(base, c1_state)
        c1_state = c1_state[:, :, :-2]
        c2_state = torch.cat([base, c1_state], -1)
        return c2_state


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

        # Node encoder: c -> attn_dim (entry + activation only; no output projection)
        # Not using build_mlp here: this encoder has no final scalar projection.
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

        # city1_scorer: concat(ctx[i], raw[i]) = attn_dim + c -> scalar
        self.city1_scorer = build_mlp(
            attn_dim + c, embed_dim, num_hidden_layers, device
        )

        # city2_scorer: concat(c1_ctx, c1_raw, ctx[i], raw[i]) = 2*attn_dim + 2*c -> scalar
        self.city2_scorer = build_mlp(
            2 * attn_dim + 2 * c, 2 * embed_dim, num_hidden_layers, device
        )

        if device != "mps":
            self.node_encoder.apply(self.init_weights)
            self.city1_scorer.apply(self.init_weights)
            nn.init.orthogonal_(self.city1_scorer[-1].weight, gain=0.01)
            self.city2_scorer.apply(self.init_weights)
            nn.init.orthogonal_(self.city2_scorer[-1].weight, gain=0.01)

    def _encode(
        self, state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Shared encoder: node encoder + positional encoding + all attention layers.

        Returns:
            ctx:      [N, D, attn_dim]  per-node context vectors
            features: [N, D, c]         raw node features (state[:, :, 1:])
            x:        [N, D, 1]         route ids       (state[:, :, :1])
        """
        x = state[:, :, :1]
        features = state[:, :, 1:]
        node_emb = self.node_encoder(features)
        node_emb = self.pos_encoder(node_emb)
        ctx = node_emb
        for attn, norm in zip(self.attention_layers, self.layer_norms):
            ctx_normed = norm(ctx)
            attn_out, _ = attn(ctx_normed, ctx_normed, ctx_normed)
            ctx = ctx + attn_out
        return ctx, features, x

    def _logits_c1(self, ctx: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        inp = torch.cat([ctx, features], dim=-1)  # [N, D, attn_dim + c]
        return self.city1_scorer(inp)[..., 0]  # [N, D]

    def _logits_c2(
        self, ctx: torch.Tensor, c1: torch.Tensor, features: torch.Tensor
    ) -> torch.Tensor:
        arange = torch.arange(ctx.shape[0], device=ctx.device)
        c1_ctx = ctx[arange, c1]  # [N, attn_dim]
        c1_raw = features[arange, c1]  # [N, c]
        c1_ctx_exp = c1_ctx[:, None, :].expand_as(ctx)  # [N, D, attn_dim]
        c1_raw_exp = c1_raw[:, None, :].expand(
            ctx.shape[0], ctx.shape[1], features.shape[-1]
        )  # [N, D, c]
        inp = torch.cat(
            [c1_ctx_exp, c1_raw_exp, ctx, features], dim=-1
        )  # [N, D, 2*attn_dim+2*c]
        return self.city2_scorer(inp)[..., 0]  # [N, D]

    def get_logits(
        self, state: torch.Tensor, action: torch.Tensor, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        ctx, features, _ = self._encode(state)
        c1 = action[:, 0]
        log_probs_c1 = torch.log(torch.softmax(self._logits_c1(ctx, features), dim=-1))
        log_probs_c2 = torch.log(
            torch.softmax(self._logits_c2(ctx, c1, features), dim=-1)
        )
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

    def sample(
        self, state: torch.Tensor, greedy: bool = False, problem=CVRP, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ctx, features, x = self._encode(state)
        n_problems = state.shape[0]

        logits = self._logits_c1(ctx, features)
        logits, _ = self._apply_mask_c1(logits, x, self.method)
        c1, log_probs_c1 = self.sample_from_logits(logits, greedy=greedy)

        logits = self._logits_c2(ctx, c1, features)
        ext_mask = (
            problem.get_action_mask(solution=x, node_pos=c1)
            if self.method == "valid"
            else None
        )
        logits, mask = self._apply_mask_c2(
            logits, c1, self.method, n_problems, external_mask=ext_mask
        )
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
        logits_c1, valid_mask_c1 = self._apply_mask_c1(logits_c1, x, self.method)

        probs_c1 = torch.softmax(logits_c1, dim=-1)
        log_probs_all_c1 = torch.log_softmax(logits_c1, dim=-1)
        p_log_p_c1 = torch.zeros_like(probs_c1)
        p_log_p_c1[valid_mask_c1] = (
            probs_c1[valid_mask_c1] * log_probs_all_c1[valid_mask_c1]
        )
        entropy_c1 = -p_log_p_c1.sum(dim=-1)
        chosen_log_prob_c1 = log_probs_all_c1.gather(1, taken_c1.view(-1, 1)).squeeze(
            -1
        )

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
        p_log_p_c2[valid_mask_c2] = (
            probs_c2[valid_mask_c2] * log_probs_all_c2[valid_mask_c2]
        )
        p_log_p_c2 = torch.nan_to_num(p_log_p_c2, nan=0.0)
        entropy_c2 = -p_log_p_c2.sum(dim=-1)
        chosen_log_prob_c2 = log_probs_all_c2.gather(1, taken_c2.view(-1, 1)).squeeze(
            -1
        )

        return chosen_log_prob_c1 + chosen_log_prob_c2, entropy_c1 + entropy_c2
