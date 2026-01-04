from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import repeat_to


class SAModel(nn.Module):
    def __init__(self, device: str = "cpu") -> None:
        super().__init__()
        self.device = device
        self.generator = torch.Generator(device=device)

    def manual_seed(self, seed: int) -> None:
        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(seed)

    def get_logits(self, state: torch.Tensor, action: torch.Tensor):
        raise NotImplementedError

    def sample(self, state: torch.Tensor, greedy: bool = False, **kwargs):
        raise NotImplementedError

    def baseline_sample(self, state: torch.Tensor, **kwargs):
        raise NotImplementedError

    @staticmethod
    def init_weights(m: nn.Module) -> None:
        if type(m) is nn.Linear:
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                m.bias.data.fill_(0.01)


class TSPActor(SAModel):
    def __init__(self, embed_dim: int, device: str) -> None:
        super().__init__(device)
        self.c1_state_dim = 7
        self.c2_state_dim = 13

        # Mean and std computation
        self.city1_net = nn.Sequential(
            nn.Linear(self.c1_state_dim, embed_dim, bias=True, device=device),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim, bias=True, device=device),
            nn.ReLU(),
            nn.Linear(embed_dim, 1, bias=False, device=device),
        )
        # Mean and std computation
        self.city2_net = nn.Sequential(
            nn.Linear(self.c2_state_dim, embed_dim, bias=True, device=device),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim, bias=True, device=device),
            nn.ReLU(),
            nn.Linear(embed_dim, 1, bias=False, device=device),
        )

        self.city1_net.apply(self.init_weights)
        self.city2_net.apply(self.init_weights)

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

    def get_logits(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        n_problems, problem_dim, _ = state.shape
        x, coords, temp = state[..., :1], state[..., 1:-1], state[..., [-1]]

        c1 = action[:, 0]
        # c2 = action[:, 1]

        # First city encoding
        coords = coords.gather(1, x.long().expand_as(coords))
        coords_prev = torch.roll(coords, 1, 1)
        coords_next = torch.roll(coords, -1, 1)
        c1_state = torch.cat([coords, coords_prev, coords_next, temp], -1)

        # City 1 net
        logits = self.city1_net(c1_state)[..., 0]
        probs = torch.softmax(logits, dim=-1)
        log_probs_c1 = torch.log(probs)

        c1_prev = (c1 - 1) % problem_dim
        c1_next = (c1 + 1) % problem_dim

        # Second city encoding: [base.prev, base, base.next,
        #                        target.prev, target, target.next]
        arange = torch.arange(n_problems)
        c1_coords = coords[arange, c1]
        c1_prev_coords = coords[arange, c1_prev]
        c1_next_coords = coords[arange, c1_next]
        base = torch.cat([c1_coords, c1_prev_coords, c1_next_coords], -1)[:, None, :]
        base = repeat_to(base, c1_state)
        c2_state = torch.cat([base, c1_state], -1)

        # City 2 net
        logits = self.city2_net(c2_state)[..., 0]
        logits[arange, c1] = -float("inf")
        logits[arange, c1_prev] = -float("inf")
        logits[arange, c1_next] = -float("inf")

        probs = torch.softmax(logits, dim=-1)
        log_probs_c2 = torch.log(probs)

        return log_probs_c1, log_probs_c2

    def baseline_sample(
        self, state: torch.Tensor, **kwargs
    ) -> Tuple[torch.Tensor, None]:
        n_problems, problem_dim, _ = state.shape

        # Sample c1 at random
        logits = torch.ones((n_problems, problem_dim), device=self.device)
        c1, _ = self.sample_from_logits(logits, one_hot=False)

        # Compute mask and sample c2
        c1_prev = (c1 - 1) % problem_dim
        c1_next = (c1 + 1) % problem_dim
        arange = torch.arange(n_problems)
        logits[arange, c1] = -float("inf")
        logits[arange, c1_prev] = -float("inf")
        logits[arange, c1_next] = -float("inf")
        c2, _ = self.sample_from_logits(logits, one_hot=False)

        # Construct action tensor and return
        action = torch.cat([c1.view(-1, 1).long(), c2.view(-1, 1).long()], dim=-1)
        return action, None

    def sample(
        self, state: torch.Tensor, greedy: bool = False, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        n_problems, problem_dim, _ = state.shape
        x, coords, temp = state[..., :1], state[..., 1:-1], state[..., [-1]]

        # First city encoding
        coords = coords.gather(1, x.long().expand_as(coords))
        coords_prev = torch.roll(coords, 1, 1)
        coords_next = torch.roll(coords, -1, 1)
        c1_state = torch.cat([coords, coords_prev, coords_next, temp], -1)

        # City 1 net
        logits = self.city1_net(c1_state)[..., 0]
        c1, log_probs_c1 = self.sample_from_logits(logits, greedy=greedy, one_hot=False)

        c1_prev = (c1 - 1) % problem_dim
        c1_next = (c1 + 1) % problem_dim

        # Second city encoding: [base.prev, base, base.next,
        #                        target.prev, target, target.next]
        arange = torch.arange(n_problems)
        c1_coords = coords[arange, c1]
        c1_prev_coords = coords[arange, c1_prev]
        c1_next_coords = coords[arange, c1_next]
        base = torch.cat([c1_coords, c1_prev_coords, c1_next_coords], -1)[:, None, :]

        base = repeat_to(base, c1_state)
        c2_state = torch.cat([base, c1_state], -1)

        # City 2 net
        logits = self.city2_net(c2_state)[..., 0]
        logits[arange, c1] = -float("inf")
        logits[arange, c1_prev] = -float("inf")
        logits[arange, c1_next] = -float("inf")
        c2, log_probs_c2 = self.sample_from_logits(logits, greedy=greedy, one_hot=False)

        # Construct action and log-probabilities
        action = torch.cat([c1.view(-1, 1).long(), c2.view(-1, 1).long()], dim=-1)
        log_probs = log_probs_c1 + log_probs_c2
        return action, log_probs[..., 0]

    def evaluate(
        self, state: torch.Tensor, action: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        n_problems, problem_dim, _ = state.shape
        x, coords, temp = state[..., :1], state[..., 1:-1], state[..., [-1]]

        c1 = action[:, 0]
        c2 = action[:, 1]

        # First city encoding
        coords = coords.gather(1, x.long().expand_as(coords))
        coords_prev = torch.roll(coords, 1, 1)
        coords_next = torch.roll(coords, -1, 1)
        c1_state = torch.cat([coords, coords_prev, coords_next, temp], -1)

        # City 1 net
        logits = self.city1_net(c1_state)[..., 0]
        probs = torch.softmax(logits, dim=-1)
        log_probs_c1 = torch.log(probs.gather(1, c1.view(-1, 1)))

        c1_prev = (c1 - 1) % problem_dim
        c1_next = (c1 + 1) % problem_dim

        # Second city encoding: [base.prev, base, base.next,
        #                        target.prev, target, target.next]
        arange = torch.arange(n_problems)
        c1_coords = coords[arange, c1]
        c1_prev_coords = coords[arange, c1_prev]
        c1_next_coords = coords[arange, c1_next]
        base = torch.cat([c1_coords, c1_prev_coords, c1_next_coords], -1)[:, None, :]
        base = repeat_to(base, c1_state)
        c2_state = torch.cat([base, c1_state], -1)

        # City 2 net
        logits = self.city2_net(c2_state)[..., 0]
        logits[arange, c1] = -float("inf")
        logits[arange, c1_prev] = -float("inf")
        logits[arange, c1_next] = -float("inf")

        probs = torch.softmax(logits, dim=-1)
        log_probs_c2 = torch.log(probs.gather(1, c2.view(-1, 1)))

        # Construct log-probabilities and return
        log_probs = log_probs_c1 + log_probs_c2
        return log_probs[..., 0]


def load_model(checkpoint_path: str, device: str = "cpu") -> TSPActor:
    """
    Loads a pre-trained TSPActor model from a checkpoint file.

    Args:
        checkpoint_path (str): The file path to the checkpoint containing the
        model's state dictionary.
        device (str, optional): The device on which to load the model.
        Defaults to "cpu".

    Returns:
        TSPActor: The loaded TSPActor model in evaluation mode.
    """
    model = TSPActor(32, device)
    model.load_state_dict(
        torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=True,
        )
    )
    model.eval()
    return model


def get_features(
    curr_sol: torch.Tensor, coords: torch.Tensor, temp: int
) -> torch.Tensor:
    """
    Generates features for the TSP actor based on the current solution,
    coordinates, and temperature.

    Args:
        curr_sol (torch.Tensor): Current solution tensor.
        coords (torch.Tensor): Coordinates of the cities.
        temp (torch.Tensor): Temperature tensor.

    Returns:
        torch.Tensor: Feature tensor for the TSP actor.
    """
    x = curr_sol.unsqueeze(-1) - 1
    coords = coords[:, 1:]
    return torch.cat([x, coords, temp * torch.ones_like(coords[..., :1])], -1)
