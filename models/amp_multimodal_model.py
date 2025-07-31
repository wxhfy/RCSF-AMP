import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)


class ActivityHead(nn.Module):
    """
    Prediction head for antimicrobial activity (main task).
    Supports post-fusion of dual-stream inputs.
    Structure: Linear -> GELU -> Dropout -> Linear
    """

    def __init__(self, input_dim=1024, hidden_dim=512, output_dim=1, dropout=0.3):
        super(ActivityHead, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.mlp(x)


class StructuralFeatureProjection(nn.Module):
    """
    Projects scalar and vector features (e.g., from a GVP) into a unified feature space,
    using numerically stable methods for handling vector features.
    """
    def __init__(self, scalar_dim=128, vector_dim=16, output_dim=512, stability_eps=1e-6):
        super(StructuralFeatureProjection, self).__init__()
        self.stability_eps = stability_eps
        
        # Process vector norms using LayerNorm for stability
        self.vector_processor = nn.Sequential(
            nn.LayerNorm(vector_dim),
            nn.Linear(vector_dim, vector_dim),
            nn.GELU(),
            nn.LayerNorm(vector_dim)
        )
        
        # Main projection network with increased depth for better expressiveness
        self.projection = nn.Sequential(
            nn.Linear(scalar_dim + vector_dim, 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Dropout(0.3),  # Add dropout for generalization
            nn.Linear(256, 512),
            nn.GELU(),
            nn.LayerNorm(512),
            nn.Dropout(0.3),
            nn.Linear(512, output_dim),
            nn.LayerNorm(output_dim)
        )
        
        # Residual connection layer (if dimensions do not match)
        self.residual_projection = nn.Linear(scalar_dim + vector_dim, output_dim)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initializes weights for improved training stability."""
        def init_linear(module):
            if isinstance(module, nn.Linear):
                # Use a small gain for initialization to prevent large initial outputs
                nn.init.xavier_uniform_(module.weight, gain=0.1)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        
        self.apply(init_linear)
    
    def _stable_vector_norm_computation(self, vector_features):
        """
        Numerically stable vector norm computation that preserves information.
        
        Args:
            vector_features: Tensor of shape [N, C, 3] representing vector features.
            
        Returns:
            Tensor of shape [N, C] with stable vector norms.
        """
        vector_norms = torch.linalg.norm(vector_features, dim=-1, ord=2)  # Shape: [N, C]
        
        # Step 1: Use soft clipping (tanh) instead of hard clipping to preserve gradients.
        vector_norms_soft = torch.tanh(vector_norms / 100.0) * 100.0
        
        # Step 2: Detect and replace non-finite values (Inf, NaN).
        valid_mask = torch.isfinite(vector_norms_soft)
        vector_norms_clean = torch.where(
            valid_mask,
            vector_norms_soft,
            torch.zeros_like(vector_norms_soft)
        )
        
        # Step 3: Add a small epsilon to very small values to prevent numerical issues.
        vector_norms_stable = torch.where(
            vector_norms_clean < self.stability_eps,
            torch.full_like(vector_norms_clean, self.stability_eps),
            vector_norms_clean
        )
        
        return vector_norms_stable

    def forward(self, scalar_features, vector_features):
        """
        Forward pass for the projection layer.
        
        Args:
            scalar_features: Tensor of shape [N, scalar_dim].
            vector_features: Tensor of shape [N, vector_dim, 3].
            
        Returns:
            Tensor of shape [N, output_dim] with the projected features.
        """
        # Compute stable norms from vector features
        vector_norms = self._stable_vector_norm_computation(vector_features)  # Shape: [N, vector_dim]
        
        # Further process the vector norms
        vector_scalar_features = self.vector_processor(vector_norms)  # Shape: [N, vector_dim]
        
        # Concatenate scalar features with the processed vector features
        concat_features = torch.cat([scalar_features, vector_scalar_features], dim=-1)
        
        # Main projection transformation
        projected_features = self.projection(concat_features)
        
        # Add residual connection
        residual = self.residual_projection(concat_features)
        projected_features = projected_features + residual
        
        return projected_features