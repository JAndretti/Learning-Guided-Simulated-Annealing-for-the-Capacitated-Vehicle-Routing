# src/model/base.py
from typing import Optional, Tuple

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


class SAModel(nn.Module):
    """Abstract base class for SA actor models."""

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
        raise NotImplementedError

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
