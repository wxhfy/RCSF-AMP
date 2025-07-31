import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class MultiHeadAttention(nn.Module):
    """
    A standard Multi-Head Attention layer.
    """
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # Project and reshape for multi-head attention
        q = self.query_proj(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key_proj(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value_proj(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute scaled dot-product attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Apply mask if provided
        if mask is not None:
            if mask.dtype != torch.bool:
                mask = mask.bool()
            attn_weights = attn_weights.masked_fill(~mask, float('-inf'))

        # Normalize with softmax and apply dropout
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention weights to values
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape, project, and return
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        attn_output = self.out_proj(attn_output)
        
        return attn_output


class MultiheadAttentionBlock(nn.Module):
    """
    A Multi-Head Attention Block (MAB), combining MHA with a feed-forward network.
    """
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiheadAttentionBlock, self).__init__()
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, query, key_value):
        # Attention, residual connection, and layer norm
        attn_output = self.attention(query, key_value, key_value)
        x = self.norm1(query + attn_output)
        
        # Feed-forward, residual connection, and layer norm
        ff_output = self.feed_forward(x)
        x = self.norm2(x + ff_output)
        
        return x


class ISAB(nn.Module):
    """
    Induced Set Attention Block (ISAB).
    Reduces complexity by having a small set of learnable inducing points attend to the input set.
    """
    def __init__(self, d_model, num_heads, num_inducing=16, dropout=0.1):
        super(ISAB, self).__init__()
        self.num_inducing = num_inducing
        # Learnable inducing points, shape [1, K, D] for broadcasting across batches
        self.inducing_points = nn.Parameter(torch.randn(1, num_inducing, d_model))
        # MAB where inducing points are the query and the input set is the key/value
        self.mab = MultiheadAttentionBlock(d_model, num_heads, dropout)

    def forward(self, x):
        """
        Forward pass for ISAB.

        Args:
            x (torch.Tensor): Input tensor (e.g., node embeddings).
                              - For a single graph: [seq_len, d_model]
                              - For a dense batch: [batch_size, seq_len, d_model]
        Returns:
            torch.Tensor: The representations of the inducing points after attending to x.
                          - For a single graph: [num_inducing, d_model]
                          - For a dense batch: [batch_size, num_inducing, d_model]
        """
        is_unbatched = x.dim() == 2
        if is_unbatched:
            x = x.unsqueeze(0) # Temporarily add a batch dimension

        batch_size = x.size(0)
        # Expand inducing points to match the batch size
        inducing_points_expanded = self.inducing_points.expand(batch_size, -1, -1)
        
        # The inducing points query the input set x
        h = self.mab(inducing_points_expanded, x)

        if is_unbatched:
            h = h.squeeze(0) # Remove temporary batch dimension

        return h


class GlobalPooling(nn.Module):
    """
    A global pooling layer using ISAB followed by mean pooling.
    Handles single graphs, dense batches, and PyG sparse batches.
    """
    def __init__(self, d_model, num_heads=8, num_inducing=16, dropout=0.1):
        super(GlobalPooling, self).__init__()
        self.isab_block = ISAB(d_model, num_heads, num_inducing, dropout)
        self.d_model = d_model

    def forward(self, x, batch: Optional[torch.Tensor] = None):
        """
        Forward pass for global pooling.

        Args:
            x (torch.Tensor): Node embeddings.
                              - PyG batch: [N_total_nodes, d_model]
                              - Dense batch: [batch_size, seq_len, d_model]
                              - Single graph: [seq_len, d_model]
            batch (torch.Tensor, optional): The batch vector from PyG [N_total_nodes].

        Returns:
            torch.Tensor: Global graph embedding(s).
                          - For batched input: [batch_size, d_model]
                          - For single graph input: [d_model]
        """
        if x.dim() == 2 and batch is not None:
            # Handle PyG sparse batch data by iterating through graphs.
            # Note: This loop-based approach can be inefficient for batches with many small graphs.
            num_graphs = batch.max().item() + 1
            output_list = []
            for i in range(num_graphs):
                graph_nodes = x[batch == i]
                if graph_nodes.numel() == 0:
                    output_list.append(torch.zeros(self.d_model, device=x.device, dtype=x.dtype))
                    continue
                
                # ISAB expects a batched input, so we add a temporary batch dimension
                inducing_repr = self.isab_block(graph_nodes.unsqueeze(0))
                # Mean pool the inducing point representations to get the global graph embedding
                pooled_graph = torch.mean(inducing_repr.squeeze(0), dim=0)
                output_list.append(pooled_graph)
            
            return torch.stack(output_list, dim=0)

        else:
            # Handle single graph or dense batch data
            inducing_repr = self.isab_block(x)
            
            # Mean pool along the inducing points dimension
            # For dense batch [B, K, D] -> [B, D]; for single graph [K, D] -> [D]
            global_embedding = torch.mean(inducing_repr, dim=-2)
            
            return global_embedding