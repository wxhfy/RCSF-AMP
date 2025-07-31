import torch
import torch.nn as nn
import torch.nn.functional as F
import math

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
    
    def forward(self, query, key_value, mask=None):
        """
        Forward pass for the CrossAttention layer.

        Args:
            query (torch.Tensor): The query features, shape [L_q, H] or [B, L_q, H].
            key_value (torch.Tensor): The key and value features, shape [L_kv, H] or [B, L_kv, H].
            mask (torch.Tensor, optional): An attention mask.

        Returns:
            torch.Tensor: The contextualized query features, with the same shape as the input query.
        """
        # Save original query for the residual connection
        residual = query
        
        # Handle 2D inputs by adding a temporary batch dimension
        is_2d = query.dim() == 2
        if is_2d:
            query, key_value, residual = query.unsqueeze(0), key_value.unsqueeze(0), residual.unsqueeze(0)

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

        # 4. Apply mask if provided
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
        attn_output = attn_output + residual
        attn_output = self.norm(attn_output)

        # Remove the temporary batch dimension if the input was 2D
        if is_2d:
            attn_output = attn_output.squeeze(0)
            
        return attn_output