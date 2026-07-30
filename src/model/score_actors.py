# src/model/score_actors.py
"""Hand-coded, non-learned SA proposal distributions.

Built for the A-bis init x proposal factorial
(docs/specs/2026-07-30-abis-stronger-baseline-design.md), which replaces the
paper's single blind-SA control with a strong non-learned one.

The scores reuse quantities the repo already computes:
  stage 1: utils.calculate_detour_features       -> Delta_i per node
  stage 2: CVRP.conditional_insertion_detour     -> insert_cost per candidate
"""

from typing import Any

import torch

from utils import calculate_detour_features

from .base import SAModel

SCORE_MODES = ("score1", "score2")


class ScoreActor(SAModel):
    """A proposal distribution built from hand-coded scores, with no parameters.

    Wears CVRPActor's `sample()` interface so `sa_test` runs unchanged and every
    factorial cell shares one Metropolis path, one cost function and one
    feasibility mask.

    Modes:
        'score1': city 1 ~ softmax(Delta_i / (alpha1 * T)), city 2 uniform over
            feasible positions. Information-matched to a checkpoint trained with
            COND_DETOUR=False, whose stage-2 net likewise sees no pair
            descriptor. Isolates hand-coded rule vs learned scorer.
        'score2': as score1, but city 2 ~ softmax(-insert_cost / (alpha2 * T)).
            Deliberately given more move-relevant information than such a
            checkpoint has.

    The 'blind' (uniform x uniform) proposal is the inherited
    `SAModel.baseline_sample`; reach it by calling sa_test with baseline=True,
    which also skips state construction entirely.

    Temperatures are coupled to the anneal, tau_k(t) = alpha_k * T(t), making
    this a heat-bath proposal at the same temperature as the acceptance test.
    High Delta_i means a badly placed node, hence worth relocating, so the sign
    is positive at stage 1 and negative at stage 2.

    Expected state: [B, L, >=2] with column 0 = node index and column 1 = the
    normalised temperature, i.e. the problem configured with every feature flag
    False except `meta`. Only those two columns are read.
    """

    def __init__(
        self,
        mode: str,
        alpha1: float,
        alpha2: float,
        init_temp: float,
        stop_temp: float,
        device: str = "cpu",
        method: str = "valid",
    ) -> None:
        """
        Args:
            mode: 'score1' or 'score2'.
            alpha1: Stage-1 temperature multiplier; tau_1(t) = alpha1 * T(t).
            alpha2: Stage-2 temperature multiplier; ignored when mode='score1'.
            init_temp: The run's INIT_TEMP, used to un-normalise the meta column.
            stop_temp: The run's STOP_TEMP, likewise.
            device: Compute device.
            method: Masking method; 'valid' uses the problem's feasibility mask,
                matching the trained checkpoints.
        """
        super().__init__(device)
        if mode not in SCORE_MODES:
            raise ValueError(f"mode must be one of {SCORE_MODES}, got {mode!r}")
        if alpha1 <= 0 or (mode == "score2" and alpha2 <= 0):
            raise ValueError(f"alphas must be > 0, got alpha1={alpha1}, alpha2={alpha2}")
        self.mode = mode
        self.alpha1 = float(alpha1)
        self.alpha2 = float(alpha2)
        self.init_temp = float(init_temp)
        self.stop_temp = float(stop_temp)
        self.method = method
        # Read by the inherited baseline_sample / _conditional_features path.
        self.cond_rank = False
        self.cond_detour = False
        self.global_context = False

    def _temperature(self, state: torch.Tensor) -> torch.Tensor:
        """Recover T(t) from the normalised `meta` column. Returns [B, 1] float32.

        Column 1 holds scale_to_unit(T, STOP_TEMP, INIT_TEMP), broadcast over
        positions, so any position recovers the same value. This is the exact
        temperature the Metropolis test uses on this step (see sa_test:318-328).
        """
        if state.size(-1) < 2:
            raise ValueError(
                f"ScoreActor needs the `meta` feature flag on (state width >= 2), "
                f"got state of shape {tuple(state.shape)}"
            )
        norm = state[:, 0, 1].to(torch.float32)  # [B]
        temp = self.stop_temp + norm * (self.init_temp - self.stop_temp)
        return temp.clamp_min(1e-8).unsqueeze(-1)  # [B, 1]

    def sample(
        self,
        state: torch.Tensor,
        greedy: bool = False,
        problem: Any = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, None]:
        """Propose a (city1, city2) position pair from the hand-coded scores.

        Args:
            state: [B, L, >=2]; column 0 = node index, column 1 = normalised temp.
            greedy: If True, take the argmax at both stages instead of sampling.
            problem: CVRP instance, for the distance matrix and feasibility mask.

        Returns:
            action:    [B, 2] positions (city1, city2)
            log_probs: [B] zeros — no PPO update consumes these at eval time
            mask:      [B, L] stage-2 validity mask, as CVRPActor.sample returns
            cond_rank: None
        """
        x = state[:, :, :1].long()  # [B, L, 1] node index per position
        n_problems, seq_len = x.shape[0], x.shape[1]
        temp = self._temperature(state)  # [B, 1]

        # --- stage 1: which node to relocate -------------------------------
        # Delta_i = d(prev,i) + d(i,succ) - d(prev,succ); high = badly placed.
        detour = calculate_detour_features(x, problem.matrix)[..., 0]  # [B, L]
        logits_c1 = detour.to(torch.float32) / (self.alpha1 * temp)
        logits_c1, _ = self._apply_mask_c1(logits_c1, x, self.method)
        c1, _ = self.sample_from_logits(logits_c1, greedy=greedy, one_hot=False)

        # --- stage 2: where to put it --------------------------------------
        ext_mask = (
            problem.get_action_mask(solution=x, node_pos=c1)
            if self.method == "valid"
            else None
        )
        if self.mode == "score2":
            # Channel 0 is insert_cost. net_delta (channel 1) differs from it by
            # removal_gain(u), constant over candidates, and softmax is
            # shift-invariant — so the two give the identical distribution.
            insert_cost = problem.conditional_insertion_detour(x, c1)[..., 0]  # [B, L]
            logits_c2 = -insert_cost.to(torch.float32) / (self.alpha2 * temp)
        else:
            logits_c2 = torch.zeros(
                n_problems, seq_len, device=logits_c1.device, dtype=torch.float32
            )
        logits_c2, mask = self._apply_mask_c2(
            logits_c2, c1, self.method, n_problems, external_mask=ext_mask
        )
        c2, _ = self.sample_from_logits(logits_c2, greedy=greedy, one_hot=False)

        action = torch.cat([c1.view(-1, 1).long(), c2.view(-1, 1).long()], dim=-1)
        log_probs = torch.zeros(n_problems, device=action.device, dtype=torch.float32)
        return action, log_probs, mask, None
