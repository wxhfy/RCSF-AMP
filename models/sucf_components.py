import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import get_laplacian, to_dense_batch

# Ensure mamba_ssm is installed
try:
    from mamba_ssm import Mamba
except ImportError:
    Mamba = None
    print("Warning: mamba-ssm is not installed. MambaLayer will not be available.")


class LaplacianPositionalEncoding(nn.Module):
    """
    Laplacian Positional Encoding module.
    Computes the first k eigenvectors of the graph Laplacian matrix as positional encodings.
    """
    def __init__(self, k=8, normalization='sym'):
        super().__init__()
        self.k = k
        self.normalization = normalization
        self.cache = {}
        
    def forward(self, data):
        """
        Computes the Laplacian Positional Encodings for a batch of graphs.

        Args:
            data (torch_geometric.data.Data): A PyG data object containing edge_index and batch attributes.
            
        Returns:
            torch.Tensor: Positional encoding features of shape [num_nodes, k].
        """
        device = data.edge_index.device
        
        # Determine the number of graphs in the batch
        if hasattr(data, 'batch') and data.batch is not None:
            batch_size = data.batch.max().item() + 1
        else: # Handle a single graph
            batch_size = 1
            data.batch = torch.zeros(data.num_nodes, dtype=torch.long, device=device)
        
        pos_encodings = []
        seq_ids = getattr(data, 'seq_id', None)
        
        # Process each graph in the batch individually
        for i in range(batch_size):
            mask = data.batch == i
            num_nodes = mask.sum().item()
            
            if num_nodes <= 1:
                pos_encodings.append(torch.zeros(num_nodes, self.k, device=device))
                continue
            
            # Attempt cache lookup using sequence identifiers when available
            cache_key = self._make_cache_key(seq_ids, i, num_nodes)
            if cache_key is not None and cache_key in self.cache:
                cached = self.cache[cache_key]
                if cached.size(0) == num_nodes:
                    pos_encodings.append(cached.to(device))
                    continue
                else:
                    self.cache.pop(cache_key, None)

            # Extract the subgraph for the current graph
            node_idx = torch.where(mask)[0]
            edge_mask = torch.isin(data.edge_index[0], node_idx) & torch.isin(data.edge_index[1], node_idx)
            
            if edge_mask.sum() == 0:
                pos_encodings.append(torch.zeros(num_nodes, self.k, device=device))
                continue
                
            # Remap edge indices to local indices for the subgraph
            local_edge_index = data.edge_index[:, edge_mask] - node_idx.min()
            
            # Compute the graph Laplacian
            L_edge_index, L_edge_weight = get_laplacian(
                local_edge_index, 
                num_nodes=num_nodes,
                normalization=self.normalization
            )
            
            try:
                # Construct the dense Laplacian matrix
                L = torch.zeros(num_nodes, num_nodes, device=device)
                L[L_edge_index[0], L_edge_index[1]] = L_edge_weight
                
                # Compute eigenvalues and eigenvectors
                _, eigenvecs = torch.linalg.eigh(L)
                
                # Select the first k eigenvectors corresponding to the smallest eigenvalues
                k_actual = min(self.k, num_nodes) 
                pos_enc = eigenvecs[:, :k_actual]
                
                # Pad with zeros if the number of eigenvectors is less than k
                if k_actual < self.k:
                    padding = torch.zeros(num_nodes, self.k - k_actual, device=device)
                    pos_enc = torch.cat([pos_enc, padding], dim=1)
                    
            except Exception:
                # Fallback to zero vectors on computation failure
                pos_enc = torch.zeros(num_nodes, self.k, device=device)
                
            pos_encodings.append(pos_enc)

            if cache_key is not None:
                self.cache[cache_key] = pos_enc.detach().cpu()
        
        return torch.cat(pos_encodings, dim=0)

    def _make_cache_key(self, seq_ids, graph_index, num_nodes):
        """Creates a cache key for the given graph if possible."""
        if seq_ids is None:
            return None

        if isinstance(seq_ids, (list, tuple)):
            if graph_index < len(seq_ids):
                return (seq_ids[graph_index], num_nodes)
            return None

        if isinstance(seq_ids, str):
            return (seq_ids, num_nodes)

        return None


class GRUGate(nn.Module):
    """
    A Gated Recurrent Unit (GRU) cell for intelligently fusing a previous state with new input.
    """
    def __init__(self, state_dim, input_dim):
        super().__init__()
        self.update_gate = nn.Linear(state_dim + input_dim, state_dim)
        self.reset_gate = nn.Linear(state_dim + input_dim, state_dim)
        self.hidden_gate = nn.Linear(state_dim + input_dim, state_dim)
        self._init_weights()
        
    def _init_weights(self):
        """Initializes GRU weights."""
        for module in [self.update_gate, self.reset_gate, self.hidden_gate]:
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)
    
    def forward(self, state, input_features):
        """
        GRU gate forward pass.

        Args:
            state (torch.Tensor): The previous state of shape [N, state_dim].
            input_features (torch.Tensor): The new input features of shape [N, input_dim].
            
        Returns:
            torch.Tensor: The updated state of shape [N, state_dim].
        """
        combined = torch.cat([state, input_features], dim=-1)
        
        update_z = torch.sigmoid(self.update_gate(combined))
        reset_r = torch.sigmoid(self.reset_gate(combined))
        
        combined_reset = torch.cat([reset_r * state, input_features], dim=-1)
        hidden_tilde = torch.tanh(self.hidden_gate(combined_reset))
        
        new_state = (1 - update_z) * state + update_z * hidden_tilde
        return new_state


class MambaLayer(nn.Module):
    """
    A Bidirectional Mamba layer for sequence modeling with linear complexity.
    """
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        if Mamba is None:
            raise ImportError("mamba-ssm is not installed. Please install it to use the MambaLayer.")
            
        self.forward_mamba = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.backward_mamba = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
            
    def forward(self, x, batch=None):
        """
        Bidirectional Mamba forward pass.

        Args:
            x (torch.Tensor): Node features [N, D] or sequence features [B, L, D].
            batch (torch.Tensor, optional): Batch index vector [N] if x represents node features.
            
        Returns:
            torch.Tensor: Bidirectionally processed features.
        """
        if batch is not None:
            # Handle PyG sparse batch format by converting to a dense tensor
            dense_x, mask = to_dense_batch(x, batch)  # [B, L, D]
            
            # Forward pass
            forward_out = self.forward_mamba(dense_x)
            
            # Backward pass
            reversed_x = torch.flip(dense_x, dims=[1])
            backward_out_reversed = self.backward_mamba(reversed_x)
            backward_out = torch.flip(backward_out_reversed, dims=[1])
            
            bidirectional_out = forward_out + backward_out
            
            # Convert back to sparse format
            return bidirectional_out[mask]
        else:
            # Handle dense tensor format directly
            forward_out = self.forward_mamba(x)
            reversed_x = torch.flip(x, dims=[1])
            backward_out_reversed = self.backward_mamba(reversed_x)
            backward_out = torch.flip(backward_out_reversed, dims=[1])
            
            return forward_out + backward_out


class TransformerLayer(nn.Module):
    """
    A Transformer encoder layer for sequence modeling as alternative to Mamba.
    """
    def __init__(self, d_model, num_heads=8, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, batch=None):
        if batch is not None:
            dense_x, mask = to_dense_batch(x, batch)
            attn_out, _ = self.self_attn(dense_x, dense_x, dense_x, key_padding_mask=~mask)
            out = self.norm1(dense_x + self.dropout(attn_out))
            ffn_out = self.linear2(self.dropout(torch.relu(self.linear1(out))))
            out = self.norm2(out + self.dropout(ffn_out))
            return out[mask]
        else:
            attn_out, _ = self.self_attn(x.unsqueeze(1), x.unsqueeze(1), x.unsqueeze(1))
            out = self.norm1(x + self.dropout(attn_out.squeeze(1)))
            ffn_out = self.linear2(self.dropout(torch.relu(self.linear1(out))))
            out = self.norm2(out + self.dropout(ffn_out))
            return out


class PLDDTGating(nn.Module):
    """Monotone-prior confidence gate that blends structure and sequence streams using pLDDT.

    The gate is anchored at AlphaFold's standard confidence threshold (pLDDT=70):
    high pLDDT residues route information from the structural stream, low pLDDT
    residues fall back to the sequence stream. Threshold and slope are learnable
    but constrained by strong priors so the gate cannot collapse to a constant.
    """

    def __init__(self, feature_dim, init_threshold=70.0, init_slope=10.0, gate_min=0.05):
        super().__init__()
        self.feature_dim = feature_dim
        self.init_threshold = float(init_threshold)
        self.gate_min = float(gate_min)
        # Learnable threshold offset, clamped via tanh to stay near AF2 convention.
        self.delta_t_raw = nn.Parameter(torch.zeros(1))
        # Learnable slope, kept positive via softplus + 5 so the sigmoid never goes too sharp.
        # Initialize raw_slope so softplus(raw_slope) + 5 ≈ init_slope.
        target_softplus = max(init_slope - 5.0, 1e-3)
        init_raw = math.log(math.expm1(target_softplus)) if target_softplus > 0 else 0.0
        self.raw_slope = nn.Parameter(torch.full((1,), float(init_raw)))
        # Per-channel modulation around the scalar gate (kept small).
        self.channel_bias = nn.Parameter(torch.zeros(feature_dim))
        # Cache for the most recent gate, exposed for diagnostics + monotonicity loss.
        self._last_gate_scalar = None

    def forward(self, struct_feats, seq_feats, plddt):
        """Fuse structure and sequence features based on pLDDT confidence.

        Args:
            struct_feats (torch.Tensor): Structure features [N, feature_dim].
            seq_feats (torch.Tensor): Sequence features [N, feature_dim].
            plddt (torch.Tensor): Raw pLDDT scores [N] or [N, 1] in [0, 100].

        Returns:
            torch.Tensor: Per-residue convex blend of the two streams.
        """
        if plddt.dim() == 1:
            plddt = plddt.unsqueeze(-1)

        delta_t = 10.0 * torch.tanh(self.delta_t_raw)
        threshold = self.init_threshold + delta_t
        slope = F.softplus(self.raw_slope) + 5.0

        scalar_gate = torch.sigmoid((plddt - threshold) / slope)
        scalar_gate = self.gate_min + (1.0 - self.gate_min) * scalar_gate

        channel_mod = 1.0 + 0.1 * torch.tanh(self.channel_bias)
        gate = scalar_gate * channel_mod
        gate = gate.clamp(min=0.0, max=1.0)

        # Cache scalar gate (per-residue, before per-channel mod) for diagnostics.
        self._last_gate_scalar = scalar_gate.detach()

        return gate * struct_feats + (1.0 - gate) * seq_feats


class SimpleConfGatedFusion(nn.Module):
    """
    Direct confidence-weighted fusion without cross-attention.
    Computes: weighted_struct = (plddt/100)*struct + (1 - plddt/100)*seq
              seq_emb_1 = GRUGate(state=seq, input_features=weighted_struct)
    """
    def __init__(self, hidden_dim):
        super().__init__()
        self.gru_gate = GRUGate(state_dim=hidden_dim, input_dim=hidden_dim)

    def forward(self, seq_emb, struct_emb, plddt):
        """
        Args:
            seq_emb: [N, hidden_dim] - sequence embedding
            struct_emb: [N, hidden_dim] - structure embedding (already processed through RGAT + pLDDT gating)
            plddt: [N] - raw pLDDT scores [0-100]
        Returns:
            seq_emb_1: [N, hidden_dim] - gated sequence embedding
        """
        if plddt.dim() == 1:
            plddt = plddt.unsqueeze(-1)

        confidence = plddt / 100.0  # [N, 1] in [0, 1]

        # Weighted blend: high pLDDT -> more structure, low pLDDT -> more sequence
        weighted_struct = confidence * struct_emb + (1.0 - confidence) * seq_emb

        # GRU-style gated update
        seq_emb_1 = self.gru_gate(state=seq_emb, input_features=weighted_struct)

        return seq_emb_1