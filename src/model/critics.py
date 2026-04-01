# src/model/critics.py
import torch
import torch.nn as nn

from .base import PositionalEncoding, build_mlp


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
        self.q_func = build_mlp(c, embed_dim, num_hidden_layers, device)
        if device != "mps":
            self.q_func.apply(self.init_weights)
            last_layer = self.q_func[-1]
            nn.init.orthogonal_(last_layer.weight, gain=1.0)
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
        num_heads: int = 4,
        device: str = "cpu",
    ) -> None:
        super().__init__()

        # Node encoder: c -> embed_dim (ReLU activations, no LeakyReLU)
        # Not using build_mlp here: architecture uses ReLU (not LeakyReLU)
        # and the final projection outputs embed_dim (not a scalar).
        layers: list[nn.Module] = []
        input_dim = c
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(input_dim, embed_dim))
            layers.append(nn.ReLU())
            input_dim = embed_dim
        layers.append(nn.Linear(embed_dim, embed_dim))
        self.node_encoder = nn.Sequential(*layers).to(device)

        self.pos_encoder = PositionalEncoding(embed_dim=embed_dim).to(device)

        self.attention_pool = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(embed_dim).to(device)
        self.kv_norm = nn.LayerNorm(embed_dim).to(device)

        # Learnable query: "What is the total value of this graph?"
        self.glimpse_query = nn.Parameter(torch.randn(1, 1, embed_dim, device=device))

        self.value_head = nn.Linear(embed_dim, 1).to(device)

        # Double application is intentional (matches original)
        self.apply(self.init_weights)
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
                m.bias.data.fill_(0.0)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass computing state values.
        Args:
            state: [Batch, Nodes, Features]
        Returns:
            value: [Batch]
        """
        x = state[:, :, 1:]
        batch_size = x.size(0)
        node_embeddings = self.node_encoder(x)
        node_embeddings = self.pos_encoder(node_embeddings)
        query = self.glimpse_query.expand(batch_size, -1, -1)
        attn_out, _ = self.attention_pool(
            self.attn_norm(query),
            self.kv_norm(node_embeddings),
            self.kv_norm(node_embeddings),
        )
        graph_embedding = query + attn_out
        value = self.value_head(graph_embedding.squeeze(1))
        return value.squeeze(-1)
