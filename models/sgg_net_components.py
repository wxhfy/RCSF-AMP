"""
SGG-Net核心组件模块
包含拉普拉斯位置编码、GRU门控和Mamba层等核心组件
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import get_laplacian, to_dense_batch
import numpy as np

# 检查是否安装了mamba-ssm
from mamba_ssm import Mamba


class LaplacianPositionalEncoding(nn.Module):
    """
    拉普拉斯位置编码模块
    计算图的拉普拉斯矩阵的前k个最小特征向量作为位置编码
    """
    def __init__(self, k=8, normalization='sym'):
        super().__init__()
        self.k = k
        self.normalization = normalization
        
    def forward(self, data):
        """
        计算拉普拉斯位置编码
        
        Args:
            data: PyG数据对象，包含edge_index和batch信息
            
        Returns:
            pos_enc: [num_nodes, k] 位置编码特征
        """
        device = data.edge_index.device
        
        # 安全地获取batch_size，处理batch为None的情况
        if hasattr(data, 'batch') and data.batch is not None:
            batch_size = data.batch.max().item() + 1
        else:
            # 如果没有batch信息，假设只有一个图
            batch_size = 1
            # 为单个图创建batch索引
            data.batch = torch.zeros(data.num_nodes, dtype=torch.long, device=device)
        
        pos_encodings = []
        
        for i in range(batch_size):
            # 获取当前图的节点掩码
            mask = data.batch == i
            num_nodes = mask.sum().item()
            
            if num_nodes <= 1:
                # 如果只有一个节点，使用零向量
                pos_enc = torch.zeros(num_nodes, self.k, device=device)
                pos_encodings.append(pos_enc)
                continue
                
            # 获取当前图的边
            node_idx = torch.where(mask)[0]
            edge_mask = torch.isin(data.edge_index[0], node_idx) & torch.isin(data.edge_index[1], node_idx)
            
            if edge_mask.sum() == 0:
                # 如果没有边，使用零向量
                pos_enc = torch.zeros(num_nodes, self.k, device=device)
                pos_encodings.append(pos_enc)
                continue
                
            # 重新映射边索引到局部索引
            local_edge_index = data.edge_index[:, edge_mask]
            local_edge_index = local_edge_index - node_idx.min()
            
            # 计算拉普拉斯矩阵
            edge_index, edge_weight = get_laplacian(
                local_edge_index, 
                num_nodes=num_nodes,
                normalization=self.normalization
            )
            
            try:
                # 构建稠密拉普拉斯矩阵
                L = torch.zeros(num_nodes, num_nodes, device=device)
                L[edge_index[0], edge_index[1]] = edge_weight
                
                # 计算特征值和特征向量
                eigenvals, eigenvecs = torch.linalg.eigh(L)
                
                # 取前k个最小特征值对应的特征向量
                k_actual = min(self.k, num_nodes - 1)  # 避免超过实际可用的特征向量数
                pos_enc = eigenvecs[:, :k_actual]
                
                # 如果k_actual < self.k，用零填充
                if k_actual < self.k:
                    padding = torch.zeros(num_nodes, self.k - k_actual, device=device)
                    pos_enc = torch.cat([pos_enc, padding], dim=1)
                    
            except Exception as e:
                # 如果计算失败，使用零向量
                pos_enc = torch.zeros(num_nodes, self.k, device=device)
                
            pos_encodings.append(pos_enc)
        
        # 连接所有图的位置编码
        return torch.cat(pos_encodings, dim=0)


class GRUGate(nn.Module):
    """
    GRU门控单元
    用于智能地融合历史状态和新输入
    """
    def __init__(self, state_dim, input_dim):
        super().__init__()
        self.state_dim = state_dim
        self.input_dim = input_dim
        
        # GRU门控参数
        self.update_gate = nn.Linear(state_dim + input_dim, state_dim)
        self.reset_gate = nn.Linear(state_dim + input_dim, state_dim)
        self.hidden_gate = nn.Linear(state_dim + input_dim, state_dim)
        
        # 初始化权重
        self._init_weights()
        
    def _init_weights(self):
        """初始化GRU权重"""
        for module in [self.update_gate, self.reset_gate, self.hidden_gate]:
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)
    
    def forward(self, state, input_features):
        """
        GRU门控前向传播
        
        Args:
            state: [N, state_dim] 历史状态
            input_features: [N, input_dim] 新输入特征
            
        Returns:
            new_state: [N, state_dim] 更新后的状态
        """
        # 拼接状态和输入
        combined = torch.cat([state, input_features], dim=-1)
        
        # 计算更新门和重置门
        update_z = torch.sigmoid(self.update_gate(combined))
        reset_r = torch.sigmoid(self.reset_gate(combined))
        
        # 计算候选隐藏状态
        combined_reset = torch.cat([reset_r * state, input_features], dim=-1)
        hidden_tilde = torch.tanh(self.hidden_gate(combined_reset))
        
        # 更新状态
        new_state = (1 - update_z) * state + update_z * hidden_tilde
        
        return new_state


class MambaLayer(nn.Module):
    """
    双向Mamba层
    提供线性复杂度的序列建模能力
    """
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        
        self.forward_mamba = Mamba(
            d_model=d_model, 
            d_state=d_state, 
            d_conv=d_conv, 
            expand=expand
        )
        self.backward_mamba = Mamba(
            d_model=d_model, 
            d_state=d_state, 
            d_conv=d_conv, 
            expand=expand
        )
              
    def forward(self, x, batch=None):
        """
        双向Mamba前向传播
        
        Args:
            x: [N, d_model] 节点特征 或 [B, L, d_model] 序列特征
            batch: [N] 批次索引 (如果x是节点特征)
            
        Returns:
            output: 双向处理后的特征
        """
        if batch is not None:
            # 处理PyG格式的节点特征
            dense_x, mask = to_dense_batch(x, batch)  # [B, L, d_model]
            forward_out = self.forward_mamba(dense_x)
            reversed_x = torch.flip(dense_x, dims=[1])
            backward_out_reversed = self.backward_mamba(reversed_x)
            backward_out = torch.flip(backward_out_reversed, dims=[1])
            bidirectional_out = forward_out + backward_out
            
            # 转换回稀疏格式
            output = bidirectional_out[mask]
            return output
        else:
            forward_out = self.forward_mamba(x)
            reversed_x = torch.flip(x, dims=[1])
            backward_out_reversed = self.backward_mamba(reversed_x)
            backward_out = torch.flip(backward_out_reversed, dims=[1])
            return forward_out + backward_out


class PLDDTGating(nn.Module):
    """
    pLDDT置信度门控模块
    根据结构预测置信度调节特征权重
    """
    def __init__(self, feature_dim):
        super().__init__()
        self.feature_dim = feature_dim
        
        # 可学习的门控参数
        self.confidence_projection = nn.Sequential(
            nn.Linear(1, feature_dim // 4),
            nn.ReLU(),
            nn.Linear(feature_dim // 4, feature_dim),
            nn.Sigmoid()
        )
    
    def forward(self, features, plddt_scores):
        """
        应用pLDDT门控
        
        Args:
            features: [N, feature_dim] 输入特征
            plddt_scores: [N] pLDDT置信度分数 (0-100)
            
        Returns:
            gated_features: [N, feature_dim] 门控后的特征
        """
        # 归一化pLDDT分数到[0,1]
        normalized_plddt = (plddt_scores / 100.0).unsqueeze(-1)  # [N, 1]
        
        # 计算门控权重
        gate_weights = self.confidence_projection(normalized_plddt)  # [N, feature_dim]
        
        # 应用门控
        gated_features = features * gate_weights
        
        return gated_features


if __name__ == "__main__":
    pass
