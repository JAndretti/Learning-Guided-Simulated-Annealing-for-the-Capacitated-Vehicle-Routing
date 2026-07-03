# src/model/actors.py
from typing import Any

import torch
import torch.nn as nn

from problem import CVRP
from utils import repeat_to

from .base import SAModel, build_mlp, pool_context


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
        global_context: bool = False,
        logit_clip: float = 0.0,
        learnable_temp: bool = False,
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
            global_context (bool): If True, concatenate a mean|max|std pooled instance
                summary g (3c extra channels) to both stages' inputs.
            logit_clip (float): 0 disables; >0 applies C*tanh(logits) clipping before
                the softmax (Kool et al., C ~ 10).
            learnable_temp (bool): If True, divide logits by a learnable temperature
                tau = exp(theta), one per stage.
        """
        super().__init__(device)
        self.method = method
        self.cond_rank = cond_rank
        self.cond_detour = cond_detour
        self.global_context = global_context
        self.init_logit_shaping(logit_clip, learnable_temp)
        # Pooled instance summary g = [mean|max|std] over nodes adds 3c dims to each stage.
        ctx = 3 * c if global_context else 0
        self.c1_state_dim = c + ctx
        # Features for city 2 include city 1's features -2 to not include meta features twice, resulting in c + c - 2.
        # +4 when cond_rank is on (edge ranks), +2 when cond_detour is on (insertion cost).
        self.c2_state_dim = (
            c * 2 - 2 + (4 if cond_rank else 0) + (2 if cond_detour else 0) + ctx
        )

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
        g = pool_context(c1_state) if self.global_context else None
        c1 = action[:, 0]  # shape: (batch_size,)

        logits_c1 = self.city1_net(self._append_context(c1_state, g))[..., 0]
        logits_c1 = self._shape_logits(logits_c1, 0)  # shape: (batch_size, problem_dim)
        probs_c1 = torch.softmax(logits_c1, dim=-1)  # shape: (batch_size, problem_dim)
        log_probs_c1 = torch.log(probs_c1)  # shape: (batch_size, problem_dim)

        cond_rank = self._conditional_features(kwargs["problem"], routes_ids, c1)
        c2_state = self._prepare_features_city2(c1_state, c1, n_problems, cond_rank)
        # c2_state shape: (batch_size, problem_dim, c2_state_dim)

        logits_c2 = self.city2_net(self._append_context(c2_state, g))[..., 0]
        logits_c2 = self._shape_logits(logits_c2, 1)  # shape: (batch_size, problem_dim)
        probs_c2 = torch.softmax(logits_c2, dim=-1)  # shape: (batch_size, problem_dim)
        log_probs_c2 = torch.log(probs_c2)  # shape: (batch_size, problem_dim)

        return log_probs_c1, log_probs_c2

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
        g = pool_context(c1_state) if self.global_context else None

        logits_c1 = self.city1_net(self._append_context(c1_state, g))[..., 0]
        logits_c1 = self._shape_logits(logits_c1, 0)  # shape: (batch_size, problem_dim)
        logits_c1, _ = self._apply_mask_c1(logits_c1, x, self.method)
        c1, log_probs_c1 = self.sample_from_logits(logits_c1, greedy=greedy, one_hot=False)

        cond_rank = self._conditional_features(problem, x, c1)
        c2_state = self._prepare_features_city2(c1_state, c1, n_problems, cond_rank)
        # c2_state shape: (batch_size, problem_dim, c2_state_dim)

        logits_c2 = self.city2_net(self._append_context(c2_state, g))[..., 0]
        logits_c2 = self._shape_logits(logits_c2, 1)  # shape: (batch_size, problem_dim)
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
        g = pool_context(c1_state) if self.global_context else None
        taken_c1 = action[:, 0]
        taken_c2 = action[:, 1]

        # Evaluate City 1
        logits_c1 = self.city1_net(self._append_context(c1_state, g))[..., 0]
        logits_c1 = self._shape_logits(logits_c1, 0)
        logits_c1, valid_mask_c1 = self._apply_mask_c1(logits_c1, x, self.method)

        probs_c1 = torch.softmax(logits_c1, dim=-1)  # shape: (batch_size, problem_dim)
        log_probs_all_c1 = torch.log_softmax(logits_c1, dim=-1)  # shape: (batch_size, problem_dim)
        p_log_p_c1 = torch.zeros_like(probs_c1)
        p_log_p_c1[valid_mask_c1] = probs_c1[valid_mask_c1] * log_probs_all_c1[valid_mask_c1]
        entropy_c1 = -p_log_p_c1.sum(dim=-1)  # shape: (batch_size,)
        chosen_log_prob_c1 = log_probs_all_c1.gather(1, taken_c1.view(-1, 1)).squeeze(-1)

        # Evaluate City 2
        c2_state = self._prepare_features_city2(c1_state, taken_c1, n_problems, cond_rank)
        logits_c2 = self.city2_net(self._append_context(c2_state, g))[..., 0]
        logits_c2 = self._shape_logits(logits_c2, 1)
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


class CVRPActorShared(SAModel):
    """
    Shared-encoder actor: encodes each node once into an embedding h_i, then
    scores the two cities with two heads over the shared embeddings.

    Stage 2 is either a concat head over [h_j | h_c1 | g? | cond?]
    (bilinear=False) or a bilinear compatibility q.k/sqrt(d) between the
    chosen node (query) and each candidate (key) (bilinear=True). The linear
    key projection is what keeps cond_rank/cond_detour supported in bilinear
    mode; a pure h^T W h would drop them.

    Like the seq actor's conditional features, cond_rank/cond_detour are not
    supported with method='rm_depot' (cond tensors keep full-N positions while
    the encoded nodes are compacted).
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
        global_context: bool = False,
        bilinear: bool = False,
        logit_clip: float = 0.0,
        learnable_temp: bool = False,
    ) -> None:
        """
        Initialize the CVRPActorShared.

        Args:
            embed_dim (int): Node embedding / hidden dimension.
            c (int): Input feature dimension per node.
            num_hidden_layers (int): Number of hidden layers in encoder and heads.
            device (str): Device to run the computations on.
            method (str): Masking method to apply ('free', 'valid', 'rm_depot').
            cond_rank (bool): Condition city-2 on bidirectional edge ranks (4 channels).
            cond_detour (bool): Condition city-2 on insertion cost (2 channels).
            global_context (bool): If True, pool node embeddings into a mean|max|std
                summary g (3*embed_dim) fed to both heads (and the bilinear query).
            bilinear (bool): If True, stage-2 logits are q.k/sqrt(d) instead of a
                concat MLP head.
            logit_clip (float): 0 disables; >0 applies C*tanh(logits) clipping.
            learnable_temp (bool): Learnable softmax temperature, one per stage.
        """
        super().__init__(device)
        self.method = method
        self.cond_rank = cond_rank
        self.cond_detour = cond_detour
        self.global_context = global_context
        self.bilinear = bilinear
        self.init_logit_shaping(logit_clip, learnable_temp)

        n_cond = (4 if cond_rank else 0) + (2 if cond_detour else 0)
        ctx = 3 * embed_dim if global_context else 0
        self.scale = embed_dim**-0.5

        self.node_encoder = build_mlp(
            c, embed_dim, num_hidden_layers, device, output_dim=embed_dim
        )
        self.c1_head = build_mlp(embed_dim + ctx, embed_dim, num_hidden_layers, device)
        if bilinear:
            self.q_proj = nn.Linear(embed_dim + ctx, embed_dim, bias=False, device=device)
            self.k_proj = nn.Linear(embed_dim + n_cond, embed_dim, bias=False, device=device)
        else:
            self.c2_head = build_mlp(
                2 * embed_dim + ctx + n_cond, embed_dim, num_hidden_layers, device
            )

        if device != "mps":
            self.node_encoder.apply(self.init_weights)
            self.c1_head.apply(self.init_weights)
            last_c1 = self.c1_head[-1]
            if isinstance(last_c1, nn.Linear):
                nn.init.orthogonal_(last_c1.weight, gain=0.01)
                if last_c1.bias is not None:
                    nn.init.constant_(last_c1.bias, 0.0)
            if bilinear:
                # Small-gain query keeps initial logits near 0 (near-uniform
                # policy), matching the 0.01 final-layer convention above.
                nn.init.orthogonal_(self.q_proj.weight, gain=0.01)
                nn.init.orthogonal_(self.k_proj.weight, gain=2**0.5)
            else:
                self.c2_head.apply(self.init_weights)
                last_c2 = self.c2_head[-1]
                if isinstance(last_c2, nn.Linear):
                    nn.init.orthogonal_(last_c2.weight, gain=0.01)
                    if last_c2.bias is not None:
                        nn.init.constant_(last_c2.bias, 0.0)

    def _encode(
        self, state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None, int, torch.Tensor]:
        """Split route ids, encode nodes once, pool the optional context.

        Args:
            state (torch.Tensor): Shape (batch_size, problem_dim, features).

        Returns:
            h: Node embeddings, shape (batch_size, problem_dim, embed_dim)
            g: Pooled context (batch_size, 3*embed_dim) or None
            n_problems: Batch size
            x: Route ids, shape (batch_size, problem_dim, 1)
        """
        n_problems, _, _ = state.shape
        x = state[:, :, :1]
        feats = state[:, :, 1:]
        if self.method == "rm_depot":
            mask = x.squeeze(-1) != 0
            feats = feats[mask].view(n_problems, -1, feats.size(-1))
        h = self.node_encoder(feats)  # shape: (batch_size, problem_dim, embed_dim)
        g = pool_context(h) if self.global_context else None
        return h, g, n_problems, x

    def _logits_c1(self, h: torch.Tensor, g: torch.Tensor | None) -> torch.Tensor:
        logits = self.c1_head(self._append_context(h, g))[..., 0]
        return self._shape_logits(logits, 0)  # shape: (batch_size, problem_dim)

    def _logits_c2(
        self,
        h: torch.Tensor,
        g: torch.Tensor | None,
        c1: torch.Tensor,
        cond: torch.Tensor | None,
    ) -> torch.Tensor:
        arange = torch.arange(h.size(0), device=h.device)
        h_c1 = h[arange, c1]  # shape: (batch_size, embed_dim)
        if self.bilinear:
            q_in = h_c1 if g is None else torch.cat([h_c1, g], dim=-1)
            k_in = h if cond is None else torch.cat([h, cond], dim=-1)
            q = self.q_proj(q_in)  # shape: (batch_size, embed_dim)
            k = self.k_proj(k_in)  # shape: (batch_size, problem_dim, embed_dim)
            logits = torch.einsum("bd,bnd->bn", q, k) * self.scale
        else:
            parts = [h, h_c1[:, None, :].expand_as(h)]
            if g is not None:
                parts.append(g[:, None, :].expand(h.size(0), h.size(1), -1))
            if cond is not None:
                parts.append(cond)
            logits = self.c2_head(torch.cat(parts, dim=-1))[..., 0]
        return self._shape_logits(logits, 1)  # shape: (batch_size, problem_dim)

    @staticmethod
    def _dist_stats(
        logits: torch.Tensor, valid_mask: torch.Tensor, taken: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Chosen log-prob and masked entropy of one stage's distribution."""
        probs = torch.softmax(logits, dim=-1)
        log_probs_all = torch.log_softmax(logits, dim=-1)
        p_log_p = torch.zeros_like(probs)
        p_log_p[valid_mask] = probs[valid_mask] * log_probs_all[valid_mask]
        p_log_p = torch.nan_to_num(p_log_p, nan=0.0)
        entropy = -p_log_p.sum(dim=-1)
        chosen = log_probs_all.gather(1, taken.view(-1, 1)).squeeze(-1)
        return chosen, entropy

    def get_logits(
        self, state: torch.Tensor, action: torch.Tensor, **kwargs: Any
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute log probabilities for given state and action.

        Args:
            state (torch.Tensor): Current state, shape: (batch_size, problem_dim, features)
            action (torch.Tensor): Given actions, shape: (batch_size, 2)

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: log_probs_c1 and log_probs_c2
                                               Shapes: both are (batch_size, problem_dim)
        """
        h, g, n_problems, x = self._encode(state)
        c1 = action[:, 0]  # shape: (batch_size,)

        logits_c1 = self._logits_c1(h, g)
        log_probs_c1 = torch.log(torch.softmax(logits_c1, dim=-1))

        cond = self._conditional_features(kwargs["problem"], x, c1)
        logits_c2 = self._logits_c2(h, g, c1, cond)
        log_probs_c2 = torch.log(torch.softmax(logits_c2, dim=-1))

        return log_probs_c1, log_probs_c2

    def sample(
        self, state: torch.Tensor, greedy: bool = False, problem: Any = CVRP, **kwargs: Any
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """
        Sample an action pair from the current state.

        Args:
            state (torch.Tensor): State tensor, shape: (batch_size, problem_dim, features)
            greedy (bool): If True, pick max logit instead of sampling.
            problem: Problem instance for obtaining action mask.

        Returns:
            actions (batch_size, 2), chosen_log_probs (batch_size,),
            mask (batch_size, problem_dim), cond_rank or None
        """
        h, g, n_problems, x = self._encode(state)

        logits_c1 = self._logits_c1(h, g)
        logits_c1, _ = self._apply_mask_c1(logits_c1, x, self.method)
        c1, log_probs_c1 = self.sample_from_logits(logits_c1, greedy=greedy, one_hot=False)

        cond_rank = self._conditional_features(problem, x, c1)
        logits_c2 = self._logits_c2(h, g, c1, cond_rank)
        ext_mask = (
            problem.get_action_mask(solution=x, node_pos=c1) if self.method == "valid" else None
        )
        logits_c2, mask = self._apply_mask_c2(
            logits_c2, c1, self.method, n_problems, external_mask=ext_mask
        )
        c2, log_probs_c2 = self.sample_from_logits(logits_c2, greedy=greedy, one_hot=False)

        action = torch.cat(
            [c1.view(-1, 1).long(), c2.view(-1, 1).long()], dim=-1
        )  # shape: (batch_size, 2)
        log_probs = log_probs_c1 + log_probs_c2
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
            cond_rank (torch.Tensor | None): Stored conditional city-2 features,
                shape (batch_size, problem_dim, n_cond); None when disabled.

        Returns:
            log_probs: Shape (batch_size,), total_entropy: Shape (batch_size,)
        """
        h, g, n_problems, x = self._encode(state)
        taken_c1 = action[:, 0]
        taken_c2 = action[:, 1]

        logits_c1 = self._logits_c1(h, g)
        logits_c1, valid_mask_c1 = self._apply_mask_c1(logits_c1, x, self.method)
        log_prob_c1, entropy_c1 = self._dist_stats(logits_c1, valid_mask_c1, taken_c1)

        logits_c2 = self._logits_c2(h, g, taken_c1, cond_rank)
        logits_c2, valid_mask_c2 = self._apply_mask_c2(
            logits_c2,
            taken_c1,
            self.method,
            n_problems,
            external_mask=mask if self.method == "valid" else None,
        )
        log_prob_c2, entropy_c2 = self._dist_stats(logits_c2, valid_mask_c2, taken_c2)

        return log_prob_c1 + log_prob_c2, entropy_c1 + entropy_c2
