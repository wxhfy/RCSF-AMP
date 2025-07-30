import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax
from torch_scatter import scatter


def custom_grouped_softmax(logits, index, dim=0):
    """
    高效的分组softmax函数，支持多头注意力的并行计算
    
    Args:
        logits: 输入logits张量 [num_edges, num_heads]
        index: 分组索引张量 [num_edges]，指示每条边属于哪个节点
        dim: softmax维度，默认为0
    
    Returns:
        经过分组softmax的权重张量，与logits形状相同
    """
    if logits.dim() == 1:
        # 一维情况，直接使用原始函数
        return softmax(logits, index)
    
    # 多维情况，使用向量化操作实现高效分组softmax
    num_edges, num_heads = logits.shape
    device = logits.device
    
    # 向量化计算：同时处理所有头，避免for循环
    # 1. 计算每个组的最大值进行数值稳定化
    max_vals = scatter(logits, index, dim=0, reduce='max')[index]  # [num_edges, num_heads]
    stable_logits = logits - max_vals
    
    # 2. 计算指数值
    exp_vals = torch.exp(stable_logits)
    
    # 3. 计算每个组的和
    sum_exp = scatter(exp_vals, index, dim=0, reduce='sum')[index]  # [num_edges, num_heads]
    
    # 4. 归一化
    result = exp_vals / (sum_exp + 1e-12)  # 避免除零
    
    return result


class RelationalGATv3Conv(MessagePassing):
    """
    关系型图注意力网络v3卷积层
    实现基于边类型的特定变换和多头注意力机制
    支持温度平滑和分阶段动态权重
    """
    
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 heads=8, 
                 dropout=0.1, 
                 edge_dim=None,
                 concat=True,
                 temperature=3.0,
                 enable_dynamic_weights=False,
                 seq_bias_dim=None):
        """
        初始化RelationalGATv3Conv
        
        参数:
            in_channels: 输入特征维度
            out_channels: 输出特征维度（每个注意力头）
            heads: 注意力头数量
            dropout: Dropout概率
            edge_dim: 边特征维度（如果有）
            concat: 是否连接多头注意力的输出（True）或取平均（False）
            temperature: 温度参数，用于平滑注意力权重分布
            enable_dynamic_weights: 是否启用动态边权重，False时使用均匀权重
            seq_bias_dim: 序列特征偏置维度（如果有）
        """
        super(RelationalGATv3Conv, self).__init__(aggr='add', node_dim=0)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.dropout = dropout
        self.edge_dim = edge_dim
        self.temperature = temperature
        self.enable_dynamic_weights = enable_dynamic_weights
        self.seq_bias_dim = seq_bias_dim
        
        # 注意力核心变换
        self.lin_query = nn.Linear(in_channels, heads * out_channels)
        self.lin_key = nn.Linear(in_channels, heads * out_channels)
        self.lin_value = nn.Linear(in_channels, heads * out_channels)
        
        # 用于保存权重信息的变量（用于日志记录）
        self._last_attention_weights = None
        self._last_edge_weights = None
        self._last_gate_weights = None
        
        # 边类型特定变换
        if edge_dim is not None:
            # 最后2维是边类型的one-hot编码
            self.edge_type_encoders = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(edge_dim - 2, 32),
                    nn.ReLU(),
                    nn.Linear(32, heads)
                ) 
                for _ in range(2)  # 为每种边类型创建一个encoder
            ])
        
        # 序列特征偏置处理器
        if seq_bias_dim is not None:
            # 简化的序列偏置处理器，避免维度爆炸
            self.seq_bias_processor = nn.Sequential(
                nn.Linear(seq_bias_dim, heads * 16),  # 减少中间层大小
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(heads * 16, heads),
                nn.Tanh()  # 限制偏置范围在[-1, 1]
            )
        
        # 注意力机制
        self.attn_dropout = nn.Dropout(dropout)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """重置参数（权重初始化）"""
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.lin_query.weight, gain=gain)
        nn.init.xavier_normal_(self.lin_key.weight, gain=gain)
        nn.init.xavier_normal_(self.lin_value.weight, gain=gain)
        
        if self.edge_dim is not None:
            for encoder in self.edge_type_encoders:
                for layer in encoder:
                    if isinstance(layer, nn.Linear):
                        nn.init.xavier_normal_(layer.weight, gain=gain)
        
        if self.seq_bias_dim is not None:
            for layer in self.seq_bias_processor:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_normal_(layer.weight, gain=gain * 0.1)  # 小初始值确保偏置温和
    
    def forward(self, x, edge_index, edge_attr=None, seq_features=None, return_attention_weights=False):
        """
        前向传播
        
        参数:
            x: 节点特征 [num_nodes, in_channels]
            edge_index: 边索引 [2, num_edges]
            edge_attr: 边属性 [num_edges, edge_dim]（如果有）
            seq_features: 序列特征 [num_nodes, seq_bias_dim]（如果有）
            return_attention_weights: 是否返回注意力权重
            
        返回:
            更新后的节点特征，可选返回注意力权重
        """
        # 计算查询、键和值
        query = self.lin_query(x).view(-1, self.heads, self.out_channels)  # [num_nodes, heads, out_channels]
        key = self.lin_key(x).view(-1, self.heads, self.out_channels)      # [num_nodes, heads, out_channels]
        value = self.lin_value(x).view(-1, self.heads, self.out_channels)  # [num_nodes, heads, out_channels]
        
        # 执行消息传递
        if seq_features is not None and hasattr(self, 'seq_bias_processor'):
            # 使用自定义方法处理序列特征偏置
            out = self._propagate_with_seq_features(
                edge_index=edge_index,
                query=query,
                key=key, 
                value=value, 
                edge_attr=edge_attr,
                seq_features=seq_features
            )
        
        # 获取注意力权重（从message方法中保存的）
        attention_weights = getattr(self, '_last_attention_weights', None)
        
        # 连接或平均多头注意力的输出
        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)
        
        if return_attention_weights:
            # 返回符合PyTorch Geometric标准的格式: (edge_index, attention_weights)
            if attention_weights is not None:
                return out, (edge_index, attention_weights)
            else:
                return out, None
        else:
            return out
    
    def _propagate_with_seq_features(self, edge_index, query, key, value, edge_attr, seq_features):
        """
        自定义的消息传递，支持序列特征偏置
        
        参数:
            edge_index: 边索引 [2, num_edges]
            query: 查询特征 [num_nodes, heads, out_channels]
            key: 键特征 [num_nodes, heads, out_channels]
            value: 值特征 [num_nodes, heads, out_channels]
            edge_attr: 边属性 [num_edges, edge_dim]
            seq_features: 序列特征 [num_nodes, seq_bias_dim]
            
        返回:
            聚合后的节点特征 [num_nodes, heads, out_channels]
        """
        # 提取边的源节点和目标节点索引
        src_nodes, dst_nodes = edge_index
        
        # 提取query, key, value对应的边特征
        query_i = query[dst_nodes]  # [num_edges, heads, out_channels]
        key_j = key[src_nodes]      # [num_edges, heads, out_channels]
        value_j = value[src_nodes]  # [num_edges, heads, out_channels]
        
        # 提取序列特征
        if seq_features is not None:
            seq_features_i = seq_features[dst_nodes]  # [num_edges, seq_bias_dim]
            seq_features_j = seq_features[src_nodes]  # [num_edges, seq_bias_dim]
        else:
            seq_features_i = None
            seq_features_j = None
        
        # 计算消息
        messages = self.message(
            query_i=query_i,
            key_j=key_j,
            value_j=value_j,
            edge_attr=edge_attr,
            seq_features_i=seq_features_i,
            seq_features_j=seq_features_j,
            index=dst_nodes,
            ptr=None,
            size_i=query.size(0)
        )
        
        # 聚合消息
        out = self.aggregate(messages, dst_nodes, dim_size=query.size(0))
        
        return out
    
    def message(self, query_i, key_j, value_j, edge_attr=None, seq_features_i=None, seq_features_j=None, index=None, ptr=None, size_i=None):
        """
        计算消息和注意力权重
        
        参数:
            query_i: 目标节点查询 [num_edges, heads, out_channels]
            key_j: 源节点键 [num_edges, heads, out_channels]
            value_j: 源节点值 [num_edges, heads, out_channels]
            edge_attr: 边属性 [num_edges, edge_dim]（如果有）
            seq_features_i: 目标节点序列特征 [num_edges, seq_bias_dim]（如果有）
            seq_features_j: 源节点序列特征 [num_edges, seq_bias_dim]（如果有）
            index: 目标节点索引
            ptr: 边批处理指针（CSR格式）
            size_i: 目标节点数量
            
        返回:
            加权消息 [num_edges, heads, out_channels]
        """
        # 1. 计算原始注意力分数: (query_i * key_j) / sqrt(out_channels)
        scale = self.out_channels ** 0.5
        attention_logits = (query_i * key_j).sum(dim=-1) / scale  # [num_edges, heads]
        
        # 2. 如果有边属性，根据边类型调整注意力分数
        if edge_attr is not None and hasattr(self, 'edge_type_encoders'):
            # 获取边类型（假设最后2维是边类型的one-hot编码）
            edge_type = torch.argmax(edge_attr[:, -2:], dim=-1)  # [num_edges]
            
            # 提取边属性（除了边类型）
            edge_features = edge_attr[:, :-2]  # [num_edges, edge_dim-2]
            
            # 为每种边类型应用特定变换
            edge_attention_modifiers = torch.zeros_like(attention_logits)  # [num_edges, heads]
            
            for edge_type_idx in range(2):  # 假设有2种边类型
                mask = (edge_type == edge_type_idx)
                if mask.sum() > 0:
                    # 计算这种边类型的注意力修正
                    type_modifier = self.edge_type_encoders[edge_type_idx](edge_features[mask])  # [num_edges_of_type, heads]
                    edge_attention_modifiers[mask] = type_modifier
            
            # 将边类型特定的修正应用到注意力logits
            attention_logits = attention_logits + edge_attention_modifiers
        
        # 3. 添加序列特征偏置
        if seq_features_i is not None and hasattr(self, 'seq_bias_processor'):
            # 更稳定的序列特征组合方式：使用concatenate而不是相加
            # seq_combined = seq_features_i + seq_features_j  # [num_edges, seq_bias_dim]
            
            # 方法1：使用L2归一化后的差异作为偏置信号
            seq_diff = F.normalize(seq_features_i, p=2, dim=-1) - F.normalize(seq_features_j, p=2, dim=-1)
            seq_bias = self.seq_bias_processor(seq_diff)  # [num_edges, heads]
            
            # 将序列偏置添加到注意力logits，但使用更小的权重
            attention_logits = attention_logits + 0.1 * seq_bias
        
        # 4. 应用温度并计算注意力权重
        if self.enable_dynamic_weights:
            # 确保温度参数有效（避免除零），但允许更小的温度值以产生更尖锐的分布
            effective_temperature = max(self.temperature, 0.01)  # 降低最小值限制
            
            # 在训练时添加少量噪声以增强泛化（但不影响温度效果）
            if self.training:
                noise_std = 0.005  # 减少噪声以更好观察温度效果
                noise = torch.randn_like(attention_logits) * noise_std
                attention_logits = attention_logits + noise
            
            # 🔥 关键修复：正确应用温度到logits，然后计算softmax
            attention_weights = custom_grouped_softmax(attention_logits / effective_temperature, index)
            
        else:
            # 使用均匀权重：创建均匀分布的注意力权重
            uniform_logits = torch.zeros_like(attention_logits)
            attention_weights = custom_grouped_softmax(uniform_logits, index)
        
        # 5. 应用Dropout
        attention_weights = self.attn_dropout(attention_weights)
        
        # 6. 保存注意力权重用于日志记录
        self._last_attention_weights = attention_weights.detach().clone()
        self._last_attention_logits = attention_logits.detach().clone()  # 保存原始logits用于调试
        self._last_temperature = effective_temperature if self.enable_dynamic_weights else 1.0  # 保存实际使用的温度
        
        # 调试信息：计算注意力分布统计
        if hasattr(self, '_debug_attention') and self._debug_attention:
            with torch.no_grad():
                attn_mean = attention_weights.mean().item()
                attn_std = attention_weights.std().item()
                attn_min = attention_weights.min().item()
                attn_max = attention_weights.max().item()
                # 计算注意力熵（衡量分布的均匀程度）
                entropy = -(attention_weights * torch.log(attention_weights + 1e-12)).sum(dim=0).mean().item()
                
                print(f"[DEBUG] 温度={effective_temperature:.3f}, 注意力统计: "
                      f"均值={attn_mean:.4f}, 标准差={attn_std:.4f}, "
                      f"范围=[{attn_min:.4f}, {attn_max:.4f}], 熵={entropy:.4f}")
        
        # 7. 计算注意力熵用于监控（不影响前向传播）
        if self.training:
            with torch.no_grad():
                # 计算每个头的平均熵
                entropy_per_head = -torch.sum(attention_weights * torch.log(attention_weights + 1e-12), dim=0)
                avg_entropy = entropy_per_head.mean().item()
                
                # 存储当前熵信息用于统计获取（避免累积过多历史数据）
                self._current_attention_entropy = avg_entropy
        
        # 8. 对值应用注意力权重
        weighted_values = value_j * attention_weights.unsqueeze(-1)  # [num_edges, heads, out_channels]
        
        return weighted_values
    
    def aggregate(self, inputs, index, ptr=None, dim_size=None):
        """聚合消息"""
        # 聚合加权的值
        return scatter(inputs, index, dim=0, dim_size=dim_size, reduce='add')
    
    def edge_updater(self, edge_index_i, edge_index_j, **kwargs):
        """边更新器 - 用于返回注意力权重"""
        if hasattr(self, '_last_attention_weights'):
            return self._last_attention_weights
        return None

    def set_dynamic_weights_enabled(self, enabled: bool):
        """设置是否启用动态边权重"""
        self.enable_dynamic_weights = enabled
        
    def update_temperature(self, new_temperature: float):
        """
        更新温度参数，确保不低于最小温度
        
        Args:
            new_temperature: 新的温度值
        """
        # 应用最小温度限制
        min_temp = getattr(self, 'min_temperature', 0.1)
        self.temperature = max(min_temp, new_temperature)
        
    def get_temperature(self) -> float:
        """获取当前温度参数"""
        return self.temperature
        
    def get_dynamic_weights_detailed_info(self):
        """获取详细的动态权重信息"""
        if hasattr(self, 'get_dynamic_weights_detailed_info'):
            return self.get_dynamic_weights_detailed_info()
        else:
            return {
                'enabled': getattr(self, 'enable_dynamic_weights', False),
                'temperature': getattr(self, 'temperature', None),
                'type': 'RGATv3'
            }
        
    def get_attention_statistics(self):
        """获取注意力权重的统计信息"""
        if not hasattr(self, '_last_attention_weights'):
            return None
        
        attention_weights = self._last_attention_weights
        
        stats = {
            'mean': attention_weights.mean().item(),
            'std': attention_weights.std().item(),
            'min': attention_weights.min().item(),
            'max': attention_weights.max().item(),
            'shape': list(attention_weights.shape)
        }
        
        # 计算熵
        entropy = -torch.sum(attention_weights * torch.log(attention_weights + 1e-12), dim=0)
        stats['entropy_mean'] = entropy.mean().item()
        stats['entropy_std'] = entropy.std().item()
        
        # 检查是否存在饱和（所有权重接近1.0或接近均匀分布）
        uniform_dist = 1.0 / attention_weights.size(0)  # 假设第0维是邻居维度
        uniform_diff = torch.abs(attention_weights - uniform_dist).mean().item()
        stats['uniformity_deviation'] = uniform_diff
        
        # 检查是否有权重过于集中
        max_weight_per_node = attention_weights.max(dim=0)[0]
        stats['max_attention_mean'] = max_weight_per_node.mean().item()
        stats['concentration_ratio'] = (max_weight_per_node > 0.8).float().mean().item()
        
        return stats
    
    def get_attention_entropy_history(self):
        """获取注意力熵的历史记录"""
        if hasattr(self, '_attention_entropy_history'):
            return self._attention_entropy_history.copy()
        return []
    
    def reset_attention_history(self):
        """重置注意力历史记录"""
        if hasattr(self, '_attention_entropy_history'):
            self._attention_entropy_history.clear()
    
    def get_attention_entropy(self, attention_weights):
        """
        计算注意力权重的熵正则化项
        
        参数:
            attention_weights: 注意力权重 [num_edges, heads]
            
        返回:
            entropy_loss: 熵损失，鼓励注意力权重分布更均匀
        """
        # 计算熵: -∑ p * log(p)
        entropy = -torch.sum(attention_weights * torch.log(attention_weights + 1e-12), dim=-1)
        
        # 返回平均熵
        return entropy.mean()
    
    def get_attention_stats(self):
        """
        获取注意力权重的统计信息，用于日志记录
        
        返回:
            dict: 包含注意力权重统计信息的字典
        """
        stats = {}
        
        if hasattr(self, '_last_attention_weights') and self._last_attention_weights is not None:
            attn_weights = self._last_attention_weights
            
            # 基础统计信息
            stats['attention'] = {
                'mean': attn_weights.mean().item(),
                'std': attn_weights.std().item(),
                'min': attn_weights.min().item(),
                'max': attn_weights.max().item(),
                'shape': list(attn_weights.shape)
            }
            
            # 添加权重分布的示例值（智能采样，避免只采样单邻居节点的1.0权重）
            flat_weights = attn_weights.flatten()
            if len(flat_weights) >= 10:
                # 统计1.0权重的比例（来自单邻居节点）
                unity_weights = flat_weights[torch.abs(flat_weights - 1.0) < 1e-6]
                non_unity_weights = flat_weights[torch.abs(flat_weights - 1.0) >= 1e-6]
                unity_ratio = len(unity_weights) / len(flat_weights)
                
                # 智能采样策略
                if len(non_unity_weights) >= 6:
                    # 如果有足够的非1.0权重，主要采样这些（更有代表性）
                    indices = torch.randperm(len(non_unity_weights))[:6]
                    sample_weights = non_unity_weights[indices]
                    # 补充少量1.0权重用于对比
                    if len(unity_weights) > 0:
                        unity_indices = torch.randperm(len(unity_weights))[:2]
                        sample_weights = torch.cat([sample_weights, unity_weights[unity_indices]])
                else:
                    # 如果非1.0权重不足，随机采样全部
                    indices = torch.randperm(len(flat_weights))[:8]
                    sample_weights = flat_weights[indices]
                
                stats['attention']['sample_values'] = sample_weights.tolist()
                stats['attention']['unity_ratio'] = float(unity_ratio)  # 单邻居权重比例
            else:
                stats['attention']['sample_values'] = flat_weights.tolist()
                stats['attention']['unity_ratio'] = 0.0
        
        # 查询、键、值权重的统计信息
        if hasattr(self, 'lin_query'):
            query_weight = self.lin_query.weight.data
            stats['query_weight'] = {
                'mean': query_weight.mean().item(),
                'std': query_weight.std().item(),
                'shape': list(query_weight.shape)
            }
        
        if hasattr(self, 'lin_key'):
            key_weight = self.lin_key.weight.data
            stats['key_weight'] = {
                'mean': key_weight.mean().item(),
                'std': key_weight.std().item(),
                'shape': list(key_weight.shape)
            }
        
        if hasattr(self, 'lin_value'):
            value_weight = self.lin_value.weight.data
            stats['value_weight'] = {
                'mean': value_weight.mean().item(),
                'std': value_weight.std().item(),
                'shape': list(value_weight.shape)
            }
        
        # 边类型权重统计
        if hasattr(self, 'edge_type_linear') and self.edge_type_linear is not None:
            edge_type_weight = self.edge_type_linear.weight.data
            stats['edge_type_weight'] = {
                'mean': edge_type_weight.mean().item(),
                'std': edge_type_weight.std().item(),
                'shape': list(edge_type_weight.shape),
                'sample_values': edge_type_weight.flatten()[:5].tolist()
            }
        
        # === 新增：边类型编码器权重统计 ===
        if hasattr(self, 'edge_type_encoders') and self.edge_type_encoders is not None:
            # 创建一个字典来存储所有边类型编码器的信息
            stats['edge_type_encoders'] = {}
            for i, encoder in enumerate(self.edge_type_encoders):
                # 获取编码器中每个线性层的信息
                encoder_info = {}
                for j, layer in enumerate(encoder):
                    if isinstance(layer, torch.nn.Linear):
                        weight = layer.weight.data
                        encoder_info[f'layer_{j}_linear'] = {
                            'mean': weight.mean().item(),
                            'std': weight.std().item(),
                            'shape': list(weight.shape),
                            'sample_values': weight.flatten()[:5].tolist()  # 记录前5个权重值作为样本
                        }
                        # 如果有偏置，也记录偏置信息
                        if layer.bias is not None:
                            bias = layer.bias.data
                            encoder_info[f'layer_{j}_bias'] = {
                                'mean': bias.mean().item(),
                                'std': bias.std().item(),
                                'shape': list(bias.shape),
                                'sample_values': bias.flatten()[:5].tolist()
                            }
                stats['edge_type_encoders'][f'type_{i}'] = encoder_info
        
        return stats
        

class RGATv3Block(nn.Module):
    """
    完整的RGATv3块，包括LayerNorm、残差连接等
    支持温度平滑和分阶段动态权重
    """
    
    def __init__(self, in_channels, out_channels, heads=8, dropout=0.3, edge_dim=None, 
                 temperature=3.0, enable_dynamic_weights=False, seq_bias_dim=None):
        """
        初始化RGATv3Block
        
        参数:
            in_channels: 输入特征维度
            out_channels: 输出特征维度（每个头）
            heads: 注意力头数量
            dropout: Dropout概率
            edge_dim: 边特征维度（如果有）
            temperature: 温度参数，用于平滑注意力权重分布
            enable_dynamic_weights: 是否启用动态边权重
            seq_bias_dim: 序列特征偏置维度（如果有）
        """
        super(RGATv3Block, self).__init__()
        
        self.norm = nn.LayerNorm(in_channels)
        
        # 如果heads*out_channels不等于in_channels，需要一个投影层进行残差连接
        need_projection = (heads * out_channels != in_channels)
        self.projection = nn.Linear(in_channels, heads * out_channels) if need_projection else None
        
        self.gatv3 = RelationalGATv3Conv(
            in_channels=in_channels,
            out_channels=out_channels,
            heads=heads,
            dropout=dropout,
            edge_dim=edge_dim,
            concat=True,
            temperature=temperature,
            enable_dynamic_weights=enable_dynamic_weights,
            seq_bias_dim=seq_bias_dim
        )
        
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, edge_index, edge_attr=None, seq_features=None):
        """
        前向传播
        
        参数:
            x: 节点特征 [num_nodes, in_channels]
            edge_index: 边索引 [2, num_edges]
            edge_attr: 边属性 [num_edges, edge_dim]（如果有）
            seq_features: 序列特征 [num_nodes, seq_bias_dim]（如果有）
            
        返回:
            更新后的节点特征 [num_nodes, heads * out_channels]
        """
        # 保存原始特征用于残差连接
        identity = x
        
        # 应用层标准化
        x = self.norm(x)
        
        # GATv3卷积 - 修复：只有当seq_features不为None时才传递该参数
        if seq_features is not None:
            x = self.gatv3(x, edge_index, edge_attr, seq_features=seq_features)
        else:
            x = self.gatv3(x, edge_index, edge_attr)
        
        # 应用激活函数
        x = self.gelu(x)
        
        # Dropout
        x = self.dropout(x)
        
        # 残差连接
        if self.projection is not None:
            identity = self.projection(identity)
        
        x = x + identity
        
        return x
    
    def set_dynamic_weights_enabled(self, enabled: bool):
        """设置是否启用动态边权重，代理到内部的gatv3组件"""
        if hasattr(self.gatv3, 'set_dynamic_weights_enabled'):
            self.gatv3.set_dynamic_weights_enabled(enabled)
    
    def get_attention_stats(self):
        """
        获取注意力权重的统计信息，用于日志记录
        代理到内部的gatv3组件
        
        返回:
            dict: 包含注意力权重统计信息的字典
        """
        # 代理到内部的gatv3组件
        if hasattr(self.gatv3, 'get_attention_stats'):
            return self.gatv3.get_attention_stats()
        else:
            return {}
    
    def get_last_attention_weights(self):
        """获取最后一次前向传播的注意力权重"""
        if hasattr(self.gatv3, '_last_attention_weights'):
            return self.gatv3._last_attention_weights
        return None