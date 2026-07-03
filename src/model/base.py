# src/model/base.py
from typing import Any, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def build_mlp(
    input_dim: int,
    embed_dim: int,
    num_hidden_layers: int,
    device: str,
    output_dim: int = 1,
) -> nn.Sequential:
    """Build an MLP: Linear+LeakyReLU entry, N hidden layers, Linear output.

    `output_dim` defaults to 1.
    """
    layers: list[nn.Module] = []
    layers.append(nn.Linear(input_dim, embed_dim, bias=True, device=device))
    layers.append(nn.LeakyReLU())
    for _ in range(num_hidden_layers):
        layers.append(nn.Linear(embed_dim, embed_dim, bias=True, device=device))
        layers.append(nn.LeakyReLU())
    layers.append(nn.Linear(embed_dim, output_dim, bias=False, device=device))
    return nn.Sequential(*layers).to(device)


def pool_context(feats: torch.Tensor) -> torch.Tensor:
    """Permutation-invariant instance summary: [B, N, d] -> [B, 3d] mean|max|std."""
    mean = feats.mean(dim=1)
    mx = feats.max(dim=1).values
    std = feats.std(dim=1, unbiased=False)
    return torch.cat([mean, mx, std], dim=-1)


class SAModel(nn.Module):
    """Abstract base class for SA actor models."""

    def __init__(self, device: str = "cpu") -> None:
        super().__init__()
        self.device = device
        self.generator = torch.Generator(device=device)

    def manual_seed(self, seed: int) -> None:
        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(seed)

    def init_logit_shaping(self, logit_clip: float = 0.0, learnable_temp: bool = False) -> None:
        """Configure optional logit clipping / learnable softmax temperature.

        Call from subclass __init__ (after nn.Module init) so the temperature
        parameter is registered. One tau per stage (0 = city 1, 1 = city 2).
        """
        self.logit_clip = float(logit_clip)
        self.learnable_temp = learnable_temp
        if learnable_temp:
            self.log_tau = nn.Parameter(torch.zeros(2, device=self.device))

    def _shape_logits(self, logits: torch.Tensor, stage: int) -> torch.Tensor:
        """Apply C*tanh clipping, then temperature tau=exp(theta).

        Must be applied at EVERY point logits are produced (sample, evaluate,
        get_logits) and BEFORE masking; any asymmetry between collection and
        PPO update corrupts the importance ratios.
        """
        if self.logit_clip > 0:
            logits = self.logit_clip * torch.tanh(logits)
        if self.learnable_temp:
            logits = logits / torch.exp(self.log_tau[stage])
        return logits

    @staticmethod
    def _append_context(feats: torch.Tensor, g: Optional[torch.Tensor]) -> torch.Tensor:
        """Broadcast-concat pooled context g [B, G] onto per-node feats [B, N, F]."""
        if g is None:
            return feats
        return torch.cat(
            [feats, g[:, None, :].expand(feats.size(0), feats.size(1), -1)], dim=-1
        )

    def get_logits(
        self, state: torch.Tensor, action: torch.Tensor, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def sample(
        self, state: torch.Tensor, greedy: bool = False, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def baseline_sample(
        self, state: torch.Tensor, problem: Any = None, **kwargs
    ) -> Tuple[torch.Tensor, None, torch.Tensor]:
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

    def _conditional_features(
        self, problem: Any, x: torch.Tensor, c1: torch.Tensor
    ) -> Optional[torch.Tensor]:
        """
        Build the conditional city-2 features that depend on the chosen city 1.

        Concatenates whichever conditional blocks are enabled into a single tensor
        (carried through the replay buffer / PPO under the 'cond_rank' slot):
          - cond_rank  -> 4 channels (bidirectional edge ranks)
          - cond_detour -> 2 channels (insert_cost, net_delta)

        Relies on `self.cond_rank` / `self.cond_detour`, set by actor subclasses.

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

    def sample_from_logits(
        self, logits: torch.Tensor, greedy: bool = False, one_hot: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample actions from logits using greedy argmax or multinomial sampling."""
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

    @staticmethod
    def _apply_mask_c1(
        logits: torch.Tensor, x: torch.Tensor, method: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply city-1 masking in-place. Returns (logits, valid_mask).

        valid_mask[i, j] is True iff position j is a valid c1 choice for
        problem i. Depot (index 0, x==0) and the last position are always
        masked when method != 'rm_depot'.
        """
        mask = torch.ones_like(logits, dtype=torch.bool)
        if method != "rm_depot":
            tmp_mask = (x != 0).squeeze(-1)
            logits[~tmp_mask] = -float("inf")
            logits[:, 0] = -float("inf")
            logits[:, -1] = -float("inf")
            mask = tmp_mask
            mask[:, 0] = False
            mask[:, -1] = False
        return logits, mask

    @staticmethod
    def _apply_mask_c2(
        logits: torch.Tensor,
        c1: torch.Tensor,
        method: str,
        n_problems: int,
        external_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply city-2 masking in-place. Returns (logits, valid_mask).

        external_mask: pre-computed feasibility mask, used only when
        method='valid'. Pass None for 'free' and 'rm_depot' methods.

        Note: for non-'valid' methods, the returned mask now marks the
        c1 index (and boundaries for 'free') as False. Previously, sample()
        returned all-True for these methods. evaluate() ignores the mask
        parameter for non-'valid' methods, so training behaviour is unchanged.
        """
        mask = torch.ones_like(logits, dtype=torch.bool)
        if method == "valid" and external_mask is not None:
            logits[~external_mask] = -float("inf")
            mask = external_mask
        else:
            arange = torch.arange(n_problems, device=logits.device)
            logits[arange, c1] = -float("inf")
            mask[arange, c1] = False
            if method == "free":
                logits[:, 0] = -float("inf")
                logits[:, -1] = -float("inf")
                mask[:, 0] = False
                mask[:, -1] = False
        return logits, mask

    @staticmethod
    def init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
