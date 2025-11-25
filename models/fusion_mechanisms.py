import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch_geometric.utils import to_dense_batch

class CrossAttention(nn.Module):
    """
    A standard cross-attention layer where a query modality attends to a key-value modality.
    """

    def __init__(self, hidden_dim=512, num_heads=8, dropout=0.1):
        """
        Initializes the CrossAttention layer.

        Args:
            hidden_dim (int): The dimensionality of the input and output features.
            num_heads (int): The number of attention heads.
            dropout (float): The dropout probability.
        """
        super(CrossAttention, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        assert self.head_dim * num_heads == hidden_dim, "hidden_dim must be divisible by num_heads"
        
        # Linear layers for query, key, and value projections
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        
        # Final output projection layer
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Dropout layers
        self.attn_dropout = nn.Dropout(dropout)
        self.output_dropout = nn.Dropout(dropout)
        
        # Layer normalization
        self.norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, query, key_value, mask=None, query_batch=None, key_value_batch=None):
        """
        Forward pass for the CrossAttention layer.

        Args:
            query (torch.Tensor): The query features, shape [L_q, H] or [B, L_q, H].
            key_value (torch.Tensor): The key and value features, shape [L_kv, H] or [B, L_kv, H].
            mask (torch.Tensor, optional): An attention mask.

        Returns:
            torch.Tensor: The contextualized query features, with the same shape as the input query.
        """
        residual = query

        query, query_mask = self._prepare_dense(query, query_batch)
        residual_dense = query
        key_value, key_mask = self._prepare_dense(
            key_value,
            key_value_batch if key_value_batch is not None else query_batch
        )

        batch_size, seq_length_q, _ = query.shape
        _, seq_length_kv, _ = key_value.shape

        # 1. Linearly project query, key, and value
        q = self.query(query)    # [B, L_q, H]
        k = self.key(key_value)      # [B, L_kv, H]
        v = self.value(key_value)    # [B, L_kv, H]

        # 2. Reshape for multi-head attention
        q = q.view(batch_size, seq_length_q, self.num_heads, self.head_dim).transpose(1, 2)  # [B, n_heads, L_q, h_dim]
        k = k.view(batch_size, seq_length_kv, self.num_heads, self.head_dim).transpose(1, 2)  # [B, n_heads, L_kv, h_dim]
        v = v.view(batch_size, seq_length_kv, self.num_heads, self.head_dim).transpose(1, 2)  # [B, n_heads, L_kv, h_dim]

        # 3. Compute attention scores (scaled dot-product)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim) # [B, n_heads, L_q, L_kv]

        # 4. Apply masks to block padding or cross-graph leakage
        if key_mask is not None:
            expanded_key_mask = key_mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, L_kv]
            attn_scores = attn_scores.masked_fill(~expanded_key_mask, float('-inf'))

        if mask is not None:
            if mask.dtype != torch.bool:
                mask = mask.bool()
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))

        # 5. Normalize scores to probabilities and apply dropout
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # 6. Apply attention weights to values
        attn_output = torch.matmul(attn_weights, v) # [B, n_heads, L_q, h_dim]

        # 7. Reshape back to original dimensions
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_length_q, self.hidden_dim) # [B, L_q, H]

        # 8. Final projection, dropout, residual connection, and normalization
        attn_output = self.out_proj(attn_output)
        attn_output = self.output_dropout(attn_output)

        # Mask padded query positions to keep zeros inactive
        if query_mask is not None:
            attn_output = attn_output.masked_fill(~query_mask.unsqueeze(-1), 0.0)

        attn_output = attn_output + residual_dense
        attn_output = self.norm(attn_output)

        return self._restore_sparse(attn_output, query_mask, residual.dim(), query_batch)

    def _prepare_dense(self, tensor, batch_index):
        """Converts sparse PyG batches to dense tensors with masks."""
        if tensor.dim() == 3:
            mask = torch.ones(
                tensor.size(0), tensor.size(1), dtype=torch.bool, device=tensor.device
            )
            return tensor, mask

        if batch_index is not None:
            dense, mask = to_dense_batch(tensor, batch_index)
            return dense, mask

        # Single graph without batch info
        dense = tensor.unsqueeze(0)
        mask = torch.ones(1, tensor.size(0), dtype=torch.bool, device=tensor.device)
        return dense, mask

    def _restore_sparse(self, tensor, mask, original_dim, batch_index):
        """Restores dense outputs back to PyG sparse formatting."""
        if original_dim == 3:
            return tensor

        if batch_index is not None and mask is not None:
            return tensor[mask]

        return tensor.squeeze(0)