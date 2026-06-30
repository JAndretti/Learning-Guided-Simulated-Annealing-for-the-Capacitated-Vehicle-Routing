# src/model/critics.py
import torch
import torch.nn as nn

from .base import build_mlp


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


class CVRPCriticDeepSets(nn.Module):
    """
    Deep-Sets critic with multi-statistic pooling and a nonlinear value head.

    Unlike CVRPCritic (which projects each node to a scalar and then averages,
    making V a linear functional of the mean node embedding), this critic:

      1. encodes each node into an `embed_dim` embedding (nonlinear MLP),
      2. pools the embeddings with mean / max / std across nodes (concatenated),
         so dispersion — clustered vs scattered instances — survives pooling,
      3. maps the pooled vector to a scalar through a nonlinear head.

    The forward signature is identical to CVRPCritic (state -> [batch_size]), so
    it is a drop-in CRITIC_MODEL option.
    """

    def __init__(
        self,
        embed_dim: int,
        c: int = 13,
        num_hidden_layers: int = 2,
        device: str | torch.device = "cpu",
    ) -> None:
        """
        Initialize the CVRPCriticDeepSets.

        Args:
            embed_dim (int): Embedding / hidden dimension size.
            c (int, optional): Input feature dimension size. Defaults to 13.
            num_hidden_layers (int, optional): Number of hidden layers in both the
                node encoder and the post-pooling head. Defaults to 2.
            device (str | torch.device, optional): Device to place the model on.
        """
        super().__init__()

        # Per-node encoder: c -> embed_dim embedding (nonlinear, output_dim=embed_dim).
        self.node_encoder = build_mlp(
            c, embed_dim, num_hidden_layers, str(device), output_dim=embed_dim
        )

        # Post-pooling head: [mean | max | std] (3 * embed_dim) -> scalar value.
        self.value_head = build_mlp(
            3 * embed_dim, embed_dim, num_hidden_layers, str(device), output_dim=1
        )

        if str(device) != "mps":
            self.node_encoder.apply(self.init_weights)
            self.value_head.apply(self.init_weights)

            # Orthogonal init on the final value projection to stabilize training.
            last_layer = self.value_head[-1]
            if isinstance(last_layer, nn.Linear):
                nn.init.orthogonal_(last_layer.weight, gain=1.0)
                if last_layer.bias is not None:
                    nn.init.constant_(last_layer.bias, 0.0)

    @staticmethod
    def init_weights(m: nn.Module) -> None:
        """Initialize Linear weights with Kaiming uniform."""
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight)
            if m.bias is not None:
                m.bias.data.fill_(0.01)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass computing state values via multi-statistic Deep-Sets pooling.

        Args:
            state (torch.Tensor): Shape [batch_size, num_nodes, total_features];
                the first feature (route id) is excluded, matching CVRPCritic.

        Returns:
            torch.Tensor: Estimated value per instance, shape [batch_size].
        """
        # Drop the leading route-id feature, as in CVRPCritic.
        x = state[:, :, 1:]  # [batch_size, num_nodes, c]

        # Encode each node independently.
        h = self.node_encoder(x)  # [batch_size, num_nodes, embed_dim]

        # Permutation-invariant pooling over the node dimension.
        mean_pool = h.mean(dim=1)  # [batch_size, embed_dim]
        max_pool = h.max(dim=1).values  # [batch_size, embed_dim]
        std_pool = h.std(dim=1, unbiased=False)  # [batch_size, embed_dim]

        pooled = torch.cat([mean_pool, max_pool, std_pool], dim=-1)  # [batch, 3*embed_dim]

        # Nonlinear head -> scalar value per instance.
        value = self.value_head(pooled)  # [batch_size, 1]
        return value.squeeze(-1)  # [batch_size]
