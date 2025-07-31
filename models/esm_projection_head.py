import torch
import torch.nn as nn

class ESMProjectionHead(nn.Module):
    """
    ESM Embedding Projection Head.
    Projects high-dimensional ESM embeddings (e.g., 2560) to a lower-dimensional space (e.g., 512).

    Input: Per-residue ESM embeddings [L, in_dim]
    Output: Projected sequence embeddings [L, out_dim]
    """

    def __init__(self, in_dim=2560, out_dim=512):
        """
        Initializes the ESM projection head.

        Args:
            in_dim (int): Input dimension (defaults to 2560 for ESM-2 3B).
            out_dim (int): Output dimension (defaults to 512).
        """
        super(ESMProjectionHead, self).__init__()

        self.projection = nn.Sequential(
            nn.Linear(in_features=in_dim, out_features=out_dim),
            nn.LayerNorm(normalized_shape=out_dim),
            nn.GELU()
        )

    def forward(self, x):
        """
        Forward pass.

        Args:
            x (torch.Tensor): Per-residue ESM embeddings of shape [L, in_dim].

        Returns:
            torch.Tensor: Projected sequence embeddings of shape [L, out_dim].
        """
        return self.projection(x)