import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax
from torch_scatter import scatter
import math

def custom_grouped_softmax(logits: torch.Tensor, index: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """
    An efficient, vectorized grouped softmax function for multi-head attention.

    Args:
        logits (torch.Tensor): Input logits tensor of shape [num_edges, num_heads].
        index (torch.Tensor): Grouping index tensor of shape [num_edges], indicating which node each edge belongs to.
        dim (int): The dimension along which to perform softmax.

    Returns:
        torch.Tensor: Group-wise softmaxed weights, same shape as logits.
    """
    if logits.dim() == 1:
        # Fallback to the standard PyG softmax for the 1D case.
        return softmax(logits, index)

    # Vectorized computation for the multi-head case to avoid loops.
    # 1. Stabilize by subtracting the max value within each group.
    max_vals = scatter(logits, index, dim=dim, reduce='max')[index]
    stable_logits = logits - max_vals
    
    # 2. Compute exponentiated values.
    exp_vals = torch.exp(stable_logits)
    
    # 3. Sum exponentiated values within each group.
    sum_exp = scatter(exp_vals, index, dim=dim, reduce='sum')[index]
    
    # 4. Normalize to get probabilities.
    return exp_vals / (sum_exp + 1e-12) # Add epsilon for numerical stability

class RelationalGATv3Conv(MessagePassing):
    """
    Relational Graph Attention Network v3 Convolutional Layer.
    Implements multi-head attention with edge-type specific transformations,
    temperature scaling, and an optional sequence-based bias.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 heads: int = 8,
                 dropout: float = 0.1,
                 edge_dim: int = None,
                 concat: bool = True,
                 temperature: float = 1.0,
                 seq_bias_dim: int = None):
        super().__init__(aggr='add', node_dim=0)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.temperature = temperature
        
        # Core attention projections
        self.lin_query = nn.Linear(in_channels, heads * out_channels)
        self.lin_key = nn.Linear(in_channels, heads * out_channels)
        self.lin_value = nn.Linear(in_channels, heads * out_channels)
        
        # Edge-type specific transformations
        if edge_dim is not None:
            # Assumes the last 2 dims of edge_attr are a one-hot encoding of edge type
            self.edge_type_encoders = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(edge_dim - 2, 32),
                    nn.ReLU(),
                    nn.Linear(32, heads)
                ) for _ in range(2)  # One encoder per edge type
            ])

        # Processor for sequence feature bias
        if seq_bias_dim is not None:
            self.seq_bias_processor = nn.Sequential(
                nn.Linear(seq_bias_dim, heads * 16),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(heads * 16, heads),
                nn.Tanh()  # Constrain bias range to [-1, 1]
            )
        
        self.attn_dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        """Initializes model weights."""
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.lin_query.weight, gain=gain)
        nn.init.xavier_normal_(self.lin_key.weight, gain=gain)
        nn.init.xavier_normal_(self.lin_value.weight, gain=gain)
        
        if hasattr(self, 'edge_type_encoders'):
            for encoder in self.edge_type_encoders:
                for layer in encoder:
                    if isinstance(layer, nn.Linear):
                        nn.init.xavier_normal_(layer.weight, gain=gain)
        
        if hasattr(self, 'seq_bias_processor'):
            for layer in self.seq_bias_processor:
                if isinstance(layer, nn.Linear):
                    # Use a smaller gain for the bias to ensure it has a gentle effect initially
                    nn.init.xavier_normal_(layer.weight, gain=gain * 0.1)

    def forward(self, x, edge_index, edge_attr=None, seq_features=None):
        """Forward pass."""
        # Project inputs into query, key, and value spaces for each head
        query = self.lin_query(x).view(-1, self.heads, self.out_channels)
        key = self.lin_key(x).view(-1, self.heads, self.out_channels)
        value = self.lin_value(x).view(-1, self.heads, self.out_channels)
        
        # Start message passing
        out = self.propagate(edge_index, query=query, key=key, value=value,
                             edge_attr=edge_attr, seq_features=seq_features)
        
        # Concatenate or average the outputs of the attention heads
        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)
            
        return out

    def message(self, query_i, key_j, value_j, edge_attr, seq_features_i, seq_features_j, index):
        """Computes messages and attention weights for each edge."""
        # 1. Calculate raw attention scores
        scale = self.out_channels ** 0.5
        attention_logits = (query_i * key_j).sum(dim=-1) / scale

        # 2. Add bias from edge-type specific features
        if edge_attr is not None and hasattr(self, 'edge_type_encoders'):
            edge_type = torch.argmax(edge_attr[:, -2:], dim=-1)
            edge_features = edge_attr[:, :-2]
            
            edge_attention_modifiers = torch.zeros_like(attention_logits)
            for i in range(len(self.edge_type_encoders)):
                mask = (edge_type == i)
                if mask.any():
                    edge_attention_modifiers[mask] = self.edge_type_encoders[i](edge_features[mask])
            
            attention_logits = attention_logits + edge_attention_modifiers

        # 3. Add bias from sequence features
        if seq_features_i is not None and hasattr(self, 'seq_bias_processor'):
            seq_diff = F.normalize(seq_features_i, p=2, dim=-1) - F.normalize(seq_features_j, p=2, dim=-1)
            seq_bias = self.seq_bias_processor(seq_diff)
            attention_logits = attention_logits + 0.1 * seq_bias

        # 4. Apply temperature and compute attention weights
        attention_weights = custom_grouped_softmax(attention_logits / self.temperature, index)
        attention_weights = self.attn_dropout(attention_weights)

        # 5. Weight values by attention and return as messages
        return value_j * attention_weights.unsqueeze(-1)

    def update_temperature(self, new_temperature: float):
        """Updates the temperature parameter."""
        self.temperature = max(0.1, new_temperature)


class RGATv3Block(nn.Module):
    """
    A complete RGATv3 block, including LayerNorm and residual connections.
    """
    def __init__(self, in_channels, out_channels, heads=8, dropout=0.3, edge_dim=None,
                 temperature=1.0, seq_bias_dim=None):
        super(RGATv3Block, self).__init__()
        
        self.norm = nn.LayerNorm(in_channels)
        
        # If input and output dimensions don't match, create a projection for the residual connection
        self.projection = None
        if (heads * out_channels != in_channels):
            self.projection = nn.Linear(in_channels, heads * out_channels)
        
        self.gatv3 = RelationalGATv3Conv(
            in_channels=in_channels,
            out_channels=out_channels,
            heads=heads,
            dropout=dropout,
            edge_dim=edge_dim,
            concat=True,
            temperature=temperature,
            seq_bias_dim=seq_bias_dim
        )
        
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_attr=None, seq_features=None):
        """Forward pass for the block."""
        identity = x
        
        # Pre-normalization
        x = self.norm(x)
        
        # GATv3 convolution
        x = self.gatv3(x, edge_index, edge_attr=edge_attr, seq_features=seq_features)
        
        x = self.gelu(x)
        x = self.dropout(x)
        
        # Residual connection
        if self.projection:
            identity = self.projection(identity)
        
        x = x + identity
        
        return x

    def update_temperature(self, new_temperature: float):
        """Proxy method to update temperature in the underlying convolution layer."""
        if hasattr(self.gatv3, 'update_temperature'):
            self.gatv3.update_temperature(new_temperature)