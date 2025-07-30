import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class CrossAttention(nn.Module):
    """
    交叉注意力层，用于一个模态关注另一个模态
    """
    
    def __init__(self, hidden_dim=512, num_heads=8, dropout=0.1):
        """
        初始化交叉注意力层
        
        参数:
            hidden_dim: 隐藏层维度
            num_heads: 注意力头数量
            dropout: Dropout概率
        """
        super(CrossAttention, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        assert self.head_dim * num_heads == hidden_dim, "hidden_dim必须能被num_heads整除"
        
        # 查询、键、值的线性层
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        
        # 输出投影
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Dropout层
        self.attn_dropout = nn.Dropout(dropout)
        self.output_dropout = nn.Dropout(dropout)
        
        # 层标准化
        self.norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, query, key_value, mask=None):
        """
        前向传播
        
        参数:
            query: 查询特征 [L, hidden_dim] 或 [batch_size, L, hidden_dim]
            key_value: 键值特征 [L, hidden_dim] 或 [batch_size, L, hidden_dim]
            mask: 注意力掩码（可选）
            
        返回:
            上下文化的查询特征，维度与输入相同
        """
        # 保存原始查询用于残差连接
        residual = query
        
        # 处理维度：如果是2D，添加batch维度
        need_squeeze = False
        if query.dim() == 2:
            need_squeeze = True
            query = query.unsqueeze(0)  # [1, L, hidden_dim]
            key_value = key_value.unsqueeze(0)  # [1, L, hidden_dim]
            residual = residual.unsqueeze(0)  # [1, L, hidden_dim]
        
        # 现在可以安全地获取批次大小
        batch_size = query.size(0)
        seq_length_q = query.size(1)
        seq_length_kv = key_value.size(1)
        
        # 线性投影
        q = self.query(query)  # [batch_size, seq_length_q, hidden_dim]
        k = self.key(key_value)  # [batch_size, seq_length_kv, hidden_dim]
        v = self.value(key_value)  # [batch_size, seq_length_kv, hidden_dim]
        
        # 将特征重塑为多头格式
        q = q.view(batch_size, seq_length_q, self.num_heads, self.head_dim).transpose(1, 2)  # [batch_size, num_heads, seq_length_q, head_dim]
        k = k.view(batch_size, seq_length_kv, self.num_heads, self.head_dim).transpose(1, 2)  # [batch_size, num_heads, seq_length_kv, head_dim]
        v = v.view(batch_size, seq_length_kv, self.num_heads, self.head_dim).transpose(1, 2)  # [batch_size, num_heads, seq_length_kv, head_dim]
        
        # 计算注意力分数
        attn_weights = torch.matmul(q, k.transpose(-2, -1))  # [batch_size, num_heads, seq_length_q, seq_length_kv]
        
        # 缩放注意力分数
        attn_weights = attn_weights / math.sqrt(self.head_dim)
        
        # 应用掩码（如果有）
        if mask is not None:
            # 确保mask是bool类型，避免PyTorch警告
            if mask.dtype != torch.bool:
                mask = mask.bool()
            attn_weights = attn_weights.masked_fill(mask == 0, float('-inf'))
        
        # Softmax归一化
        attn_weights = F.softmax(attn_weights, dim=-1)
        
        # Dropout
        attn_weights = self.attn_dropout(attn_weights)
        
        # 应用注意力权重
        attn_output = torch.matmul(attn_weights, v)  # [batch_size, num_heads, seq_length_q, head_dim]
        
        # 重塑为原始格式
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_length_q, self.hidden_dim)  # [batch_size, seq_length_q, hidden_dim]
        
        # 最终线性投影
        attn_output = self.out_proj(attn_output)
        
        # Dropout
        attn_output = self.output_dropout(attn_output)
        
        # 添加残差连接
        attn_output = attn_output + residual
        
        # 应用层标准化
        attn_output = self.norm(attn_output)
        
        # 如果需要，去除假的批量维度
        if need_squeeze:
            attn_output = attn_output.squeeze(0)  # [L, hidden_dim]
        
        return attn_output


