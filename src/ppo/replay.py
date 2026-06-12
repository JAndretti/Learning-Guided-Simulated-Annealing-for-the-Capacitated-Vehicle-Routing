from collections import deque
from typing import Deque

import torch
from tensordict import TensorDict, lazy_stack


class ReplayBuffer:
    def __init__(
        self, capacity: int, device: torch.device = torch.device("cpu")
    ) -> None:
        self.capacity = capacity
        self.device = device
        self._memory: Deque[TensorDict] = deque([], maxlen=capacity)

    def push(
        self,
        state: torch.Tensor,
        mask: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        old_log_probs: torch.Tensor,
        done: bool = False,
        cond_rank: torch.Tensor | None = None,
    ) -> None:
        assert len(self._memory) < self.capacity, (
            f"ReplayBuffer full ({self.capacity} steps) — increase capacity or clear before pushing"
        )
        data = {
            "state": state.detach().clone().to(self.device),
            "mask": mask.detach().clone().to(self.device),
            "action": action.detach().clone().to(self.device),
            "reward": reward.detach().clone().to(self.device),
            "old_log_probs": old_log_probs.detach().clone().to(self.device),
            "done": torch.tensor(done, dtype=torch.bool, device=self.device),
        }
        # Only present when conditional-rank conditioning is enabled. The flag is constant
        # within a run, so every stacked TensorDict shares the same key set.
        if cond_rank is not None:
            data["cond_rank"] = cond_rank.detach().clone().to(self.device)
        td = TensorDict(data, batch_size=[])
        self._memory.append(td)

    def mark_terminal(self) -> None:
        if self._memory:
            self._memory[-1]["done"] = torch.tensor(True, dtype=torch.bool, device=self.device)

    def get_all_ordered(self) -> TensorDict:
        return lazy_stack(list(self._memory)).to_tensordict()

    def clear(self) -> None:
        self._memory.clear()

    def __len__(self) -> int:
        return len(self._memory)
