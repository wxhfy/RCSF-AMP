import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter
from typing import Tuple, Optional, Union
import math
import logging

logger = logging.getLogger(__name__)


class GVP(nn.Module):
    """
    几何向量感知器（Geometric Vector Perceptron）
    处理标量和向量特征，带有数值稳定性保护
    """

    def __init__(self,
                 scalar_input_dim: int,
                 scalar_output_dim: int,
                 vector_input_dim: int,  # 向量特征的通道数
                 vector_output_dim: int,  # 向量特征的通道数
                 activation: nn.Module = nn.SiLU(),
                 vector_gate: bool = True,
                 stability_eps: float = 1e-6):
        super(GVP, self).__init__()

        # 标量部分输入维度 = scalar_input_dim (来自输入标量) + vector_input_dim (来自输入向量的范数)
        self.scalar_linear = nn.Linear(scalar_input_dim + vector_input_dim, scalar_output_dim)

        # 向量部分权重由标量特征生成
        # 输出 vector_output_dim 个权重矩阵，每个矩阵大小为 vector_input_dim (将 vector_input_dim 个输入向量通道映射到1个输出向量通道)
        # 所以总输出是 vector_output_dim * vector_input_dim 个权重值
        self.vector_linear = nn.Linear(scalar_input_dim, vector_output_dim * vector_input_dim)

        self.vector_gate = vector_gate
        if vector_gate:
            # 门控也由 h_scalar (scalar_input_dim + vector_input_dim) 控制
            self.vector_gate_linear = nn.Linear(scalar_input_dim + vector_input_dim, vector_output_dim)

        self.activation = activation
        self.vector_input_dim = vector_input_dim
        self.vector_output_dim = vector_output_dim
        self.stability_eps = stability_eps
        
        # 数值稳定性组件
        self.vector_layernorm = nn.LayerNorm(vector_input_dim) if vector_input_dim > 0 else None
        
        # 权重初始化以提高稳定性
        self._init_weights()

    def _init_weights(self):
        """权重初始化以提高数值稳定性"""
        # 对线性层使用Xavier初始化，并添加小的正则化
        nn.init.xavier_uniform_(self.scalar_linear.weight, gain=0.1)
        nn.init.constant_(self.scalar_linear.bias, 0)
        
        nn.init.xavier_uniform_(self.vector_linear.weight, gain=0.1)
        nn.init.constant_(self.vector_linear.bias, 0)
        
        if self.vector_gate:
            nn.init.xavier_uniform_(self.vector_gate_linear.weight, gain=0.1)
            nn.init.constant_(self.vector_gate_linear.bias, 0)

    def _stable_vector_norm(self, vectors):
        """计算稳定的向量范数，避免梯度爆炸"""
        # 使用稳定的向量范数计算，避免除零和梯度爆炸
        norms = torch.norm(vectors, dim=-1, p=2)
        
        # 软裁剪：使用tanh来平滑地限制范数
        stable_norms = torch.tanh(norms / 10.0) * 10.0
        
        # 确保数值稳定性
        stable_norms = torch.where(
            torch.isfinite(stable_norms),
            stable_norms,
            torch.zeros_like(stable_norms)
        )
        
        return stable_norms

    def _normalize_vectors(self, vectors):
        """对向量进行稳定的归一化，保持方向信息"""
        # 计算向量的范数
        norms = torch.norm(vectors, dim=-1, keepdim=True)
        
        # 避免除零：如果范数太小，保持原向量
        safe_norms = torch.clamp(norms, min=self.stability_eps)
        
        # 归一化向量
        normalized_vectors = vectors / safe_norms
        
        # 软缩放：对于非常大的向量，逐渐减少其幅度
        scale_factor = torch.tanh(norms / 100.0)
        scaled_vectors = normalized_vectors * scale_factor * torch.clamp(norms, max=100.0)
        
        # 处理NaN/Inf
        stable_vectors = torch.where(
            torch.isfinite(scaled_vectors),
            scaled_vectors,
            torch.zeros_like(scaled_vectors)
        )
        
        return stable_vectors

    def forward(self, scalar_features: torch.Tensor, vector_features: Optional[torch.Tensor]) -> Tuple[
        torch.Tensor, torch.Tensor]:
        # scalar_features: [..., S_in]
        # vector_features: [..., V_in, 3] or None

        if vector_features is not None and self.vector_input_dim > 0:
            # 对向量特征进行数值稳定性处理
            vector_features_stable = self._normalize_vectors(vector_features)
            
            # 计算稳定的向量范数
            vector_norms = self._stable_vector_norm(vector_features_stable)  # [..., V_in]
            
            # 如果有LayerNorm，对范数进行归一化
            if self.vector_layernorm is not None:
                vector_norms = self.vector_layernorm(vector_norms)
            
            # 将标量特征和向量范数拼接
            h_scalar = torch.cat([scalar_features, vector_norms], dim=-1)  # [..., S_in + V_in]
        else:
            h_scalar = scalar_features
            vector_features_stable = torch.zeros(*scalar_features.shape[:-1], self.vector_input_dim, 3,
                                               device=scalar_features.device,
                                               dtype=scalar_features.dtype)

        # 标量输出
        scalar_out = self.activation(self.scalar_linear(h_scalar))  # [..., S_out]

        # 向量输出
        if self.vector_output_dim > 0:
            # 获取向量变换权重 [batch_size, V_out * V_in]
            vector_weights = self.vector_linear(scalar_features)  # [..., V_out * V_in]
            # 重塑为 [..., V_out, V_in]
            vector_weights = vector_weights.view(*vector_weights.shape[:-1], self.vector_output_dim, self.vector_input_dim)

            # 对权重进行软正则化而不是硬裁剪
            vector_weights = torch.tanh(vector_weights)  # 软限制在[-1, 1]范围内

            # 应用线性变换: [..., V_out, V_in] × [..., V_in, 3] -> [..., V_out, 3]
            vector_out = torch.einsum('...ij,...jk->...ik', vector_weights, vector_features_stable)
            
            # 对输出向量进行稳定性处理
            vector_out = self._normalize_vectors(vector_out)

            # 应用门控
            if self.vector_gate:
                gates = torch.sigmoid(self.vector_gate_linear(h_scalar))  # [..., V_out]
                gates = gates.unsqueeze(-1)  # [..., V_out, 1]
                vector_out = gates * vector_out  # [..., V_out, 3]
        else:
            vector_out = torch.zeros(*scalar_features.shape[:-1], self.vector_output_dim, 3,
                                    device=scalar_features.device,
                                    dtype=scalar_features.dtype)

        return scalar_out, vector_out


class RelationalGVPConv(nn.Module):
    """
    关系式GVP卷积层，按照GVP-GNN原文实现
    直接实现为普通模块，自己处理消息传递和聚合
    带有数值稳定性保护
    """
    def __init__(self,
                 node_scalar_dim: int,
                 node_vector_dim: int, # 这是矢量通道数
                 edge_scalar_dim: int,
                 hidden_scalar_dim: int,
                 hidden_vector_dim: int,
                 output_scalar_dim: int,
                 output_vector_dim: int,
                 stability_eps: float = 1e-6):
        super(RelationalGVPConv, self).__init__()
        
        self.message_gvp = GVP(
            scalar_input_dim=node_scalar_dim + edge_scalar_dim,
            scalar_output_dim=hidden_scalar_dim,
            vector_input_dim=node_vector_dim + 1,  # 节点向量 + 边向量
            vector_output_dim=hidden_vector_dim,
            stability_eps=stability_eps
        )

        self.update_gvp = GVP(
            scalar_input_dim=hidden_scalar_dim + node_scalar_dim, # 聚合消息 + 自身
            scalar_output_dim=output_scalar_dim,
            vector_input_dim=hidden_vector_dim + node_vector_dim, # 聚合消息 + 自身
            vector_output_dim=output_vector_dim,
            stability_eps=stability_eps
        )

        self.node_vector_dim = node_vector_dim

    def forward(self, x_s: torch.Tensor, x_v: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: torch.Tensor, edge_vector: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        GVP消息传递，同时处理序列边和结构边
        
        Args:
            x_s: 节点标量特征 [N, node_scalar_dim]
            x_v: 节点矢量特征 [N, node_vector_dim, 3]
            edge_index: 边索引 [2, E]
            edge_attr: 边标量特征 [E, edge_scalar_dim]
            edge_vector: 边矢量特征 [E, 1, 3]
        
        Returns:
            out_s: 输出标量特征 [N, output_scalar_dim]
            out_v: 输出矢量特征 [N, output_vector_dim, 3]
        """
        N = x_s.size(0)
        
        # 验证输入格式
        if x_v.size(1) != self.node_vector_dim:
            raise ValueError(f"Input node_vector channels {x_v.size(1)} != expected {self.node_vector_dim}")

        # 消息生成阶段
        row, col = edge_index  # row: source (j), col: target (i)
        
        # 获取源节点特征
        x_j_s = x_s[row]  # [E, node_scalar_dim]
        x_j_v = x_v[row]  # [E, node_vector_dim, 3]
        

        # 准备消息GVP的输入：拼接源节点特征和边特征
        msg_scalar_input = torch.cat([x_j_s, edge_attr], dim=-1)  # [E, node_scalar_dim + edge_scalar_dim]
        
        msg_vector_input = torch.cat([x_j_v, edge_vector], dim=1)  # [E, node_vector_dim + 1, 3]

        # 通过消息GVP生成消息
        msg_s, msg_v = self.message_gvp(msg_scalar_input, msg_vector_input)

        # 消息聚合阶段：使用scatter进行sum聚合
        msg_s_agg = scatter(msg_s, col, dim=0, dim_size=N, reduce='sum')  # [N, hidden_scalar_dim]
        msg_v_agg = scatter(msg_v, col, dim=0, dim_size=N, reduce='sum')  # [N, hidden_vector_dim, 3]

        # 节点更新阶段：拼接聚合消息和原始节点特征
        update_scalar_input = torch.cat([msg_s_agg, x_s], dim=-1)  # [N, hidden_scalar_dim + node_scalar_dim]
        update_vector_input = torch.cat([msg_v_agg, x_v], dim=1)    # [N, hidden_vector_dim + node_vector_dim, 3]

        # 通过更新GVP生成最终输出
        out_s, out_v = self.update_gvp(update_scalar_input, update_vector_input)

        return out_s, out_v


class RGVPEncoder(nn.Module):
    """
    关系式GVP编码器，带有增强的数值稳定性保护
    """
    def __init__(self,
                 node_input_scalar_dim=22,
                 node_input_vector_dim=1,
                 edge_input_scalar_dim=10,
                 hidden_scalar_dim=128,
                 hidden_vector_dim=16,
                 output_scalar_dim=128,
                 output_vector_dim=16,
                 num_layers=1,
                 dropout=0.1,
                 stability_eps=1e-6):
        super(RGVPEncoder, self).__init__()

        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        self.scalar_norms = nn.ModuleList()
        self.vector_norms = nn.ModuleList()  # 添加向量特征的LayerNorm
        self.stability_eps = stability_eps

        current_scalar_dim = node_input_scalar_dim
        current_vector_dim = node_input_vector_dim

        for i in range(num_layers):
            s_out_dim = output_scalar_dim if i == num_layers - 1 else hidden_scalar_dim
            v_out_dim = output_vector_dim if i == num_layers - 1 else hidden_vector_dim

            self.convs.append(RelationalGVPConv(
                node_scalar_dim=current_scalar_dim,
                node_vector_dim=current_vector_dim,
                edge_scalar_dim=edge_input_scalar_dim,
                hidden_scalar_dim=hidden_scalar_dim,
                hidden_vector_dim=hidden_vector_dim,
                output_scalar_dim=s_out_dim,
                output_vector_dim=v_out_dim,
                stability_eps=stability_eps  # 传递稳定性参数
            ))
            self.scalar_norms.append(nn.LayerNorm(s_out_dim))
            # 为向量特征添加归一化（对每个通道的范数进行归一化）
            self.vector_norms.append(nn.LayerNorm(v_out_dim))

            current_scalar_dim = s_out_dim
            current_vector_dim = v_out_dim

        self.dropout_layer = nn.Dropout(dropout)
        
        # 添加输出稳定性层
        self.output_stabilizer = VectorStabilizer(
            vector_dim=output_vector_dim,
            stability_eps=stability_eps
        )

    def _apply_vector_layernorm(self, vector_features, layer_norm):
        """
        对向量特征应用LayerNorm
        对每个向量通道的范数进行归一化
        """
        # 计算每个向量的范数 [N, V, 3] -> [N, V]
        vector_norms = torch.norm(vector_features, dim=-1, p=2)
        
        # 应用LayerNorm到范数上
        normalized_norms = layer_norm(vector_norms)  # [N, V]
        
        # 重新缩放向量特征
        # 避免除零
        safe_norms = torch.clamp(vector_norms, min=self.stability_eps)
        scaling_factor = normalized_norms / safe_norms
        
        # 应用缩放 [N, V] -> [N, V, 1] -> [N, V, 3]
        scaled_vectors = vector_features * scaling_factor.unsqueeze(-1)
        
        return scaled_vectors

    def forward(self, x_s_in, x_v_in, edge_index, edge_attr, edge_vector):
        """
        GVP编码器前向传播，带有增强的数值稳定性
        
        Args:
            x_s_in: 节点标量特征 [N, node_scalar_dim]
            x_v_in: 节点矢量特征 [N, node_vector_dim, 3]
            edge_index: 边索引 [2, E]
            edge_attr: 边标量特征 [E, edge_scalar_dim]
            edge_vector: 边矢量特征 [E, 1, 3]
        
        Returns:
            scalar_h: 输出标量特征 [N, output_scalar_dim]
            vector_h: 输出矢量特征 [N, output_vector_dim, 3]
        """
        scalar_h = x_s_in
        vector_h = x_v_in

        for i in range(self.num_layers):
            scalar_prev, vector_prev = scalar_h, vector_h

            # GVP消息传递
            scalar_h_next, vector_h_next = self.convs[i](
                scalar_h,    # 对应 x_s
                vector_h,    # 对应 x_v
                edge_index,  # 对应 edge_index
                edge_attr,   # 对应 edge_attr
                edge_vector  # 对应 edge_vector
            )

            # 标量特征的层归一化
            scalar_h_norm = self.scalar_norms[i](scalar_h_next)
            
            # 向量特征的层归一化
            vector_h_norm = self._apply_vector_layernorm(vector_h_next, self.vector_norms[i])

            # Dropout
            scalar_h_dropped = self.dropout_layer(scalar_h_norm)
            vector_h_dropped = self.dropout_layer(vector_h_norm)

            # 稳定的残差连接
            if scalar_h_dropped.shape == scalar_prev.shape:
                scalar_h = scalar_prev + scalar_h_dropped
            else:
                scalar_h = scalar_h_dropped

            if vector_h_dropped.shape == vector_prev.shape:
                vector_h = vector_prev + vector_h_dropped
            else:
                vector_h = vector_h_dropped

        # 最终输出稳定化
        vector_h = self.output_stabilizer(vector_h)
        
        return scalar_h, vector_h


class VectorStabilizer(nn.Module):
    """
    向量特征稳定化器，用于最终输出的稳定性保护
    """
    def __init__(self, vector_dim, stability_eps=1e-6):
        super(VectorStabilizer, self).__init__()
        self.vector_dim = vector_dim
        self.stability_eps = stability_eps
        
        # 学习的稳定化参数
        self.scale_factor = nn.Parameter(torch.ones(1))
        self.max_norm = nn.Parameter(torch.full((1,), 100.0))
        
    def forward(self, vector_features):
        """
        对向量特征进行稳定化处理
        
        Args:
            vector_features: [N, V, 3] 向量特征
            
        Returns:
            stabilized_vectors: [N, V, 3] 稳定化后的向量特征
        """
        # 计算向量范数
        norms = torch.norm(vector_features, dim=-1, keepdim=True)  # [N, V, 1]
        
        # 软裁剪：使用可学习的最大范数
        soft_max_norm = torch.abs(self.max_norm)  # 确保正数
        scale = torch.tanh(norms / soft_max_norm)
        
        # 应用软缩放
        stabilized_vectors = vector_features * scale * torch.abs(self.scale_factor)
        
        # 最终的NaN/Inf清理
        stabilized_vectors = torch.where(
            torch.isfinite(stabilized_vectors),
            stabilized_vectors,
            torch.zeros_like(stabilized_vectors)
        )
        
        return stabilized_vectors