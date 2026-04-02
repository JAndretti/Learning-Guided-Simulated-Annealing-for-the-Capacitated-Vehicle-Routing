# src/model/critics.py
import torch
import torch.nn as nn

from .base import PositionalEncoding, build_mlp


class CVRPCritic(nn.Module):
    """
    Critic network for CVRP that estimates the state value.
    It takes node features, processes them through an MLP to get a scalar value per node,
    and then pools these values (using mean) to output a single value for the entire graph.
    """

    def __init__(
        self,
        embed_dim: int,
        c: int = 13,
        num_hidden_layers: int = 2,
        device: str | torch.device = "cpu",
    ) -> None:
        """
        Initialize the CVRPCritic.

        Args:
            embed_dim (int): Hidden dimension size for the MLP layers.
            c (int, optional): Input feature dimension size. Defaults to 13.
            num_hidden_layers (int, optional): Number of hidden layers in MLP. Defaults to 2.
            device (str | torch.device, optional): Device to place the model on. Defaults to "cpu".
        """
        super().__init__()

        # Build an MLP mapping from 'c' input features to a scalar (since it's a value function)
        # Using build_mlp which outputs a single dimension at the end.
        self.q_func = build_mlp(c, embed_dim, num_hidden_layers, str(device))

        # Initialize weights if not using MPS device
        if str(device) != "mps":
            self.q_func.apply(self.init_weights)

            # Use orthogonal initialization for the last layer to stabilize training
            last_layer = self.q_func[-1]
            if isinstance(last_layer, nn.Linear):
                nn.init.orthogonal_(last_layer.weight, gain=1.0)
                if last_layer.bias is not None:
                    nn.init.constant_(last_layer.bias, 0.0)

    @staticmethod
    def init_weights(m: nn.Module) -> None:
        """
        Initialize weights using Kaiming uniform initialization.

        Args:
            m (nn.Module): The module to initialize.
        """
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight)
            if m.bias is not None:
                m.bias.data.fill_(0.01)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass computing state values.

        Args:
            state (torch.Tensor): The input state tensor describing the CVRP formulation.
                                  Shape: [batch_size, num_nodes, features+1] (or similar)

        Returns:
            torch.Tensor: The estimated value for each problem instance.
                          Shape: [batch_size]
        """
        # state shape: [batch_size, num_nodes, total_features]
        batch_size, num_nodes, dim = state.shape

        # Exclude the first feature (often an identifier or padding) from the input
        # state shape becomes: [batch_size, num_nodes, total_features - 1]
        state = state[:, :, 1:]

        # Process node features through the MLP independently for each node
        # The output of q_func is squeezed or viewed to be [batch_size, num_nodes]
        # Intermediate shape inside q_func: [batch_size, num_nodes, 1]
        q_values = self.q_func(state).view(batch_size, num_nodes)

        # Pool the individual node values via mean to obtain a graph-level value
        # q_values shape becomes: [batch_size]
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
        device: str | torch.device = "cpu",
    ) -> None:
        """
        Initialize the CVRPCriticAttention.

        Args:
            embed_dim (int): Embedding dimension size.
            c (int, optional): Input feature dimension size. Defaults to 13.
            num_hidden_layers (int, optional): Number of hidden layers in node encoder MLP. Defaults to 2.
            num_heads (int, optional): Number of attention heads for pooling. Defaults to 4.
            device (str | torch.device, optional): Device to place the model on. Defaults to "cpu".
        """
        super().__init__()

        # Node encoder: c -> embed_dim (ReLU activations, no LeakyReLU)
        # It maps raw node features to high-dimensional embeddings
        layers: list[nn.Module] = []
        input_dim = c
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(input_dim, embed_dim))
            layers.append(nn.ReLU())
            input_dim = embed_dim
        layers.append(nn.Linear(embed_dim, embed_dim))

        self.node_encoder = nn.Sequential(*layers).to(device)

        # Positional encoding adds spatial/sequence information to embeddings
        self.pos_encoder = PositionalEncoding(embed_dim=embed_dim).to(device)

        # Multi-head attention used for aggregating node embeddings
        self.attention_pool = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True,
        )

        # Layer normalization for queries, keys, and values
        self.attn_norm = nn.LayerNorm(embed_dim).to(device)
        self.kv_norm = nn.LayerNorm(embed_dim).to(device)

        # Learnable query parameter: acts as a summary token for the entire graph
        # Shape: [1, 1, embed_dim]
        self.glimpse_query = nn.Parameter(torch.randn(1, 1, embed_dim, device=device))

        # Final linear projection to get a scalar value from the aggregated graph embedding
        self.value_head = nn.Linear(embed_dim, 1).to(device)

        # Initialize weights 
        self.apply(self.init_weights)
        # Specific orthogonal initialization for the final value head
        if str(device) != "mps":
            nn.init.orthogonal_(self.value_head.weight, gain=1.0)
            if self.value_head.bias is not None:
                nn.init.constant_(self.value_head.bias, 0.0)

    @staticmethod
    def init_weights(m: nn.Module) -> None:
        """
        Initialize weights using Kaiming uniform initialization.

        Args:
            m (nn.Module): The module to initialize.
        """
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass computing state values via attention pooling.

        Args:
            state (torch.Tensor): The input state tensor describing the CVRP formulation.
                                  Shape: [batch_size, num_nodes, features+1]

        Returns:
            torch.Tensor: The estimated value for each problem instance.
                          Shape: [batch_size]
        """
        # Extract features (excluding the first feature as before)
        # x shape: [batch_size, num_nodes, features]
        x = state[:, :, 1:]

        # Get the batch size for expanding the learnable query
        batch_size = x.size(0)

        # Encode raw node features into embeddings
        # node_embeddings shape: [batch_size, num_nodes, embed_dim]
        node_embeddings = self.node_encoder(x)

        # Add positional encodings to distinguish node positions/identities
        # node_embeddings shape remains: [batch_size, num_nodes, embed_dim]
        node_embeddings = self.pos_encoder(node_embeddings)

        # Expand the learnable query to match the batch size
        # query shape: [batch_size, 1, embed_dim]
        query = self.glimpse_query.expand(batch_size, -1, -1)

        # Apply multi-head attention pooling
        # query: [batch_size, 1, embed_dim]
        # keys/values: [batch_size, num_nodes, embed_dim]
        # attn_out shape: [batch_size, 1, embed_dim]
        attn_out, _ = self.attention_pool(
            self.attn_norm(query),
            self.kv_norm(node_embeddings),
            self.kv_norm(node_embeddings),
        )

        # Add a residual connection around the attention layer
        # graph_embedding shape: [batch_size, 1, embed_dim]
        graph_embedding = query + attn_out

        # Squeeze the sequence dimension and map to a scalar value
        # graph_embedding.squeeze(1) shape: [batch_size, embed_dim]
        # value shape: [batch_size, 1]
        value = self.value_head(graph_embedding.squeeze(1))

        # Squeeze the final singleton dimension to get a 1D tensor of values
        # Output shape: [batch_size]
        return value.squeeze(-1)
