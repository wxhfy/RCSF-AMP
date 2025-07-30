import torch
import torch.nn as nn



import logging

logger = logging.getLogger(__name__)


class ActivityHead(nn.Module):
    """
    抗菌活性预测头 (主任务) - 支持双流后融合
    结构: Linear(input_dim, hidden_dim) -> GELU -> Linear(hidden_dim, 1)
    """

    def __init__(self, input_dim=1024, hidden_dim=512, output_dim=1, dropout=0.3):  # input_dim默认为hidden_dim*2
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
    结构特征投影层，将GVP输出的标量和向量特征投影到统一的特征空间
    采用数值稳定的向量处理方法
    """
    def __init__(self, scalar_dim=128, vector_dim=16, output_dim=512, stability_eps=1e-6):
        super(StructuralFeatureProjection, self).__init__()
        self.stability_eps = stability_eps
        
        # 向量特征处理：使用LayerNorm而不是简单的范数计算
        self.vector_processor = nn.Sequential(
            nn.LayerNorm(vector_dim),
            nn.Linear(vector_dim, vector_dim),
            nn.GELU(),
            nn.LayerNorm(vector_dim)
        )
        
        # 主投影网络：更深层的网络以提高表达能力
        self.projection = nn.Sequential(
            nn.Linear(scalar_dim + vector_dim, 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Dropout(0.3),  # 添加dropout提高泛化能力
            nn.Linear(256, 512),
            nn.GELU(),
            nn.LayerNorm(512),
            nn.Dropout(0.3),
            nn.Linear(512, output_dim),
            nn.LayerNorm(output_dim)
        )
        
        # 残差连接层（如果维度匹配）
        self.residual_projection = None
        if scalar_dim + vector_dim != output_dim:
            self.residual_projection = nn.Linear(scalar_dim + vector_dim, output_dim)
        
        self._init_weights()
    
    def _init_weights(self):
        """权重初始化以提高稳定性"""
        def init_linear(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.1)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        
        self.apply(init_linear)
    
    def _stable_vector_norm_computation(self, vector_features):
        """
        稳定的向量范数计算，保留更多信息
        
        Args:
            vector_features: [N, C, 3] 向量特征
            
        Returns:
            vector_norms: [N, C] 稳定的向量范数
        """
        # 计算向量范数
        vector_norms = torch.linalg.norm(vector_features, dim=-1, ord=2)  # [N, C]
        
        # 方法1：使用软裁剪而不是硬裁剪，保留更多梯度信息
        # 使用tanh进行软限制，而不是clamp的硬限制
        vector_norms_soft = torch.tanh(vector_norms / 100.0) * 100.0
        
        # 方法2：对极端值进行检测和处理
        # 检测并替换非有限值（Inf, NaN）
        valid_mask = torch.isfinite(vector_norms_soft)
        vector_norms_clean = torch.where(
            valid_mask,
            vector_norms_soft,
            torch.zeros_like(vector_norms_soft)
        )
        
        # 方法3：使用稳定的数值范围
        # 对于非常小的值，添加小的epsilon以避免数值问题
        vector_norms_stable = torch.where(
            vector_norms_clean < self.stability_eps,
            torch.full_like(vector_norms_clean, self.stability_eps),
            vector_norms_clean
        )
        
        return vector_norms_stable

    def forward(self, scalar_features, vector_features):
        """
        前向传播
        
        Args:
            scalar_features: [N, scalar_dim] 标量特征
            vector_features: [N, vector_dim, 3] 向量特征
            
        Returns:
            projected_features: [N, output_dim] 投影后的特征
        """
        # 计算稳定的向量范数
        vector_norms = self._stable_vector_norm_computation(vector_features)  # [N, vector_dim]
        
        # 对向量范数进行进一步处理
        vector_scalar_features = self.vector_processor(vector_norms)  # [N, vector_dim]
        
        # 拼接标量特征和处理后的向量特征
        concat_features = torch.cat([scalar_features, vector_scalar_features], dim=-1)  # [N, scalar_dim + vector_dim]
        
        # 主要的投影变换
        projected_features = self.projection(concat_features)  # [N, output_dim]
        
        # 添加残差连接（如果可能）
        if self.residual_projection is not None:
            residual = self.residual_projection(concat_features)
            projected_features = projected_features + residual
        
        return projected_features

