import torch
import torch.nn as nn
from typing import Tuple, Optional
from torch_scatter import scatter
class GVP(nn.Module):
    """
    Geometric Vector Perceptron (GVP) layer.
    Processes scalar and vector features with added numerical stability guards.
    """

    def __init__(self,
                 scalar_input_dim: int,
                 scalar_output_dim: int,
                 vector_input_dim: int,
                 vector_output_dim: int,
                 activation: nn.Module = nn.SiLU(),
                 vector_gate: bool = True,
                 stability_eps: float = 1e-6):
        super(GVP, self).__init__()

        # The input to the scalar projection includes input scalars and norms of input vectors.
        self.scalar_linear = nn.Linear(scalar_input_dim + vector_input_dim, scalar_output_dim)

        # The weights for the vector transformation are dynamically generated from scalar features.
        self.vector_linear = nn.Linear(scalar_input_dim, vector_output_dim * vector_input_dim)

        self.vector_gate = vector_gate
        if vector_gate:
            # The gate is also controlled by the combined scalar features.
            self.vector_gate_linear = nn.Linear(scalar_input_dim + vector_input_dim, vector_output_dim)

        self.activation = activation
        self.vector_input_dim = vector_input_dim
        self.vector_output_dim = vector_output_dim
        self.stability_eps = stability_eps
        
        # Component for numerical stability of vector features.
        self.vector_layernorm = nn.LayerNorm(vector_input_dim) if vector_input_dim > 0 else None
        
        self._init_weights()

    def _init_weights(self):
        """Initializes weights for better numerical stability."""
        # Use deterministic initialization for reproducibility
        torch.nn.init.xavier_uniform_(self.scalar_linear.weight, gain=0.1)
        torch.nn.init.constant_(self.scalar_linear.bias, 0)
        
        torch.nn.init.xavier_uniform_(self.vector_linear.weight, gain=0.1)
        torch.nn.init.constant_(self.vector_linear.bias, 0)
        
        if self.vector_gate:
            torch.nn.init.xavier_uniform_(self.vector_gate_linear.weight, gain=0.1)
            torch.nn.init.constant_(self.vector_gate_linear.bias, 0)

    def _stable_vector_norm(self, vectors: torch.Tensor) -> torch.Tensor:
        """Calculates vector norms in a numerically stable manner."""
        norms = torch.norm(vectors, dim=-1, p=2)
        # Use tanh for soft clipping to smoothly limit large norms and preserve gradients.
        stable_norms = torch.tanh(norms / 10.0) * 10.0
        return torch.where(torch.isfinite(stable_norms), stable_norms, torch.zeros_like(stable_norms))

    def _normalize_vectors(self, vectors: torch.Tensor) -> torch.Tensor:
        """Normalizes vectors stably, preserving their direction."""
        norms = torch.norm(vectors, dim=-1, keepdim=True)
        # Avoid division by zero by clamping small norms.
        safe_norms = torch.clamp(norms, min=self.stability_eps)
        normalized_vectors = vectors / safe_norms
        
        # Clean up any potential NaN/Inf values.
        return torch.where(torch.isfinite(normalized_vectors), normalized_vectors, torch.zeros_like(normalized_vectors))

    def forward(self, scalar_features: torch.Tensor, vector_features: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        # Expected shapes: scalar_features [..., S_in], vector_features [..., V_in, 3]
        if vector_features is not None and self.vector_input_dim > 0:
            vector_norms = self._stable_vector_norm(vector_features)
            if self.vector_layernorm is not None:
                vector_norms = self.vector_layernorm(vector_norms)
            
            h_scalar = torch.cat([scalar_features, vector_norms], dim=-1)
        else:
            # If no vector features are provided, pad the scalar input.
            padding = torch.zeros(*scalar_features.shape[:-1], self.vector_input_dim,
                                  device=scalar_features.device, dtype=scalar_features.dtype)
            h_scalar = torch.cat([scalar_features, padding], dim=-1)
            vector_features = torch.zeros(*scalar_features.shape[:-1], self.vector_input_dim, 3,
                                          device=scalar_features.device, dtype=scalar_features.dtype)

        scalar_out = self.activation(self.scalar_linear(h_scalar))

        if self.vector_output_dim > 0:
            vector_weights = self.vector_linear(scalar_features)
            vector_weights = vector_weights.view(*vector_weights.shape[:-1], self.vector_output_dim, self.vector_input_dim)
            
            # Apply linear transformation: [..., V_out, V_in] x [..., V_in, 3] -> [..., V_out, 3]
            vector_out = torch.einsum('...ij,...jk->...ik', vector_weights, vector_features)
            vector_out = self._normalize_vectors(vector_out)

            if self.vector_gate:
                gates = torch.sigmoid(self.vector_gate_linear(h_scalar))
                vector_out = gates.unsqueeze(-1) * vector_out
        else:
            vector_out = torch.zeros(*scalar_out.shape[:-1], self.vector_output_dim, 3,
                                     device=scalar_out.device, dtype=scalar_out.dtype)

        return scalar_out, vector_out


class RelationalGVPConv(nn.Module):
    """
    A Relational GVP Convolution layer, implemented as a standard module that
    manually handles message passing and aggregation.
    """
    def __init__(self,
                 node_scalar_dim: int, node_vector_dim: int,
                 edge_scalar_dim: int,
                 hidden_scalar_dim: int, hidden_vector_dim: int,
                 output_scalar_dim: int, output_vector_dim: int,
                 stability_eps: float = 1e-6):
        super(RelationalGVPConv, self).__init__()
        
        # GVP for message creation
        self.message_gvp = GVP(
            scalar_input_dim=node_scalar_dim + edge_scalar_dim,
            scalar_output_dim=hidden_scalar_dim,
            vector_input_dim=node_vector_dim + 1,  # Node vectors + edge vectors
            vector_output_dim=hidden_vector_dim,
            stability_eps=stability_eps
        )

        # GVP for node updates
        self.update_gvp = GVP(
            scalar_input_dim=hidden_scalar_dim + node_scalar_dim, # Aggregated messages + self
            scalar_output_dim=output_scalar_dim,
            vector_input_dim=hidden_vector_dim + node_vector_dim, # Aggregated messages + self
            vector_output_dim=output_vector_dim,
            stability_eps=stability_eps
        )

    def forward(self, x_s: torch.Tensor, x_v: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: torch.Tensor, edge_vector: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        
        num_nodes = x_s.size(0)
        src_nodes, dst_nodes = edge_index

        # --- Message Creation Stage ---
        # Gather source node features for each edge
        x_j_s, x_j_v = x_s[src_nodes], x_v[src_nodes]

        # Prepare message GVP input by concatenating source node and edge features
        msg_scalar_input = torch.cat([x_j_s, edge_attr], dim=-1)
        msg_vector_input = torch.cat([x_j_v, edge_vector], dim=1)
        msg_s, msg_v = self.message_gvp(msg_scalar_input, msg_vector_input)

        # --- Aggregation Stage ---
        # Aggregate messages at destination nodes using scatter sum
        msg_s_agg = scatter(msg_s, dst_nodes, dim=0, dim_size=num_nodes, reduce='sum')
        msg_v_agg = scatter(msg_v, dst_nodes, dim=0, dim_size=num_nodes, reduce='sum')

        # --- Node Update Stage ---
        # Prepare update GVP input by concatenating aggregated messages and original node features
        update_scalar_input = torch.cat([msg_s_agg, x_s], dim=-1)
        update_vector_input = torch.cat([msg_v_agg, x_v], dim=1)
        out_s, out_v = self.update_gvp(update_scalar_input, update_vector_input)

        return out_s, out_v


class RGVPEncoder(nn.Module):
    """
    A multi-layer Relational GVP Encoder with enhanced numerical stability.
    """
    def __init__(self,
                 node_input_scalar_dim=22, node_input_vector_dim=1,
                 edge_input_scalar_dim=10,
                 hidden_scalar_dim=128, hidden_vector_dim=16,
                 output_scalar_dim=128, output_vector_dim=16,
                 num_layers=1, dropout=0.1, stability_eps=1e-6):
        super(RGVPEncoder, self).__init__()

        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        self.scalar_norms = nn.ModuleList()
        self.vector_norms = nn.ModuleList()
        self.stability_eps = stability_eps

        current_scalar_dim, current_vector_dim = node_input_scalar_dim, node_input_vector_dim

        for i in range(num_layers):
            s_out = output_scalar_dim if i == num_layers - 1 else hidden_scalar_dim
            v_out = output_vector_dim if i == num_layers - 1 else hidden_vector_dim

            self.convs.append(RelationalGVPConv(
                node_scalar_dim=current_scalar_dim, node_vector_dim=current_vector_dim,
                edge_scalar_dim=edge_input_scalar_dim,
                hidden_scalar_dim=hidden_scalar_dim, hidden_vector_dim=hidden_vector_dim,
                output_scalar_dim=s_out, output_vector_dim=v_out,
                stability_eps=stability_eps
            ))
            self.scalar_norms.append(nn.LayerNorm(s_out))
            self.vector_norms.append(nn.LayerNorm(v_out))

            current_scalar_dim, current_vector_dim = s_out, v_out

        self.dropout_layer = nn.Dropout(dropout)
        
    def _apply_vector_layernorm(self, vector_features: torch.Tensor, layer_norm: nn.LayerNorm) -> torch.Tensor:
        """Applies LayerNorm to the norms of vector features."""
        vector_norms = torch.norm(vector_features, dim=-1, p=2)
        normalized_norms = layer_norm(vector_norms)
        
        # Rescale original vectors by the ratio of new-norm to old-norm
        safe_norms = torch.clamp(vector_norms, min=self.stability_eps)
        scaling_factor = normalized_norms / safe_norms
        
        return vector_features * scaling_factor.unsqueeze(-1)

    def forward(self, x_s_in, x_v_in, edge_index, edge_attr, edge_vector):
        scalar_h, vector_h = x_s_in, x_v_in

        for i in range(self.num_layers):
            scalar_prev, vector_prev = scalar_h, vector_h

            scalar_h, vector_h = self.convs[i](scalar_h, vector_h, edge_index, edge_attr, edge_vector)
            
            # Apply dropout and normalization
            scalar_h = self.dropout_layer(self.scalar_norms[i](scalar_h))
            vector_h = self.dropout_layer(self._apply_vector_layernorm(vector_h, self.vector_norms[i]))

            # Residual connection
            if scalar_h.shape == scalar_prev.shape:
                scalar_h = scalar_prev + scalar_h
            if vector_h.shape == vector_prev.shape:
                vector_h = vector_prev + vector_h
                
        return scalar_h, vector_h