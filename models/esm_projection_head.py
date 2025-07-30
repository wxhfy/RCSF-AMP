import torch
import torch.nn as nn

class ESMProjectionHead(nn.Module):
    """
    ESM嵌入投影层，将高维ESM嵌入（2560维）投影到较低维度空间（512维）
    
    输入: 每个残基的ESM嵌入 [L, 2560]
    输出: 投影后的序列嵌入 Eseq′ [L, 512]
    """
    
    def __init__(self, in_dim=2560, out_dim=512):
        """
        初始化ESM投影头
        
        参数:
            in_dim: 输入维度（默认为ESM2的2560维）
            out_dim: 输出维度（默认为512维）
        """
        super(ESMProjectionHead, self).__init__()
        
        self.projection = nn.Sequential(
            nn.Linear(in_features=in_dim, out_features=out_dim),
            nn.LayerNorm(normalized_shape=out_dim),
            nn.GELU()
        )
    
    def forward(self, x):
        """
        前向传播
        
        参数:
            x: 每个残基的ESM嵌入 [L, 2560]
            
        返回:
            投影后的序列嵌入 Eseq′ [L, 512]
        """
        return self.projection(x)