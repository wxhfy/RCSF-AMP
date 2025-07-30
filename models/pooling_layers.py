from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    """
    多头注意力层，ISAB的核心组件
    """

    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model必须能被num_heads整除"
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
        
        # 安全检查：输入验证
        if torch.isnan(query).any() or torch.isinf(query).any():
            raise ValueError("Query包含NaN或Inf值")
        if torch.isnan(key).any() or torch.isinf(key).any():
            raise ValueError("Key包含NaN或Inf值")
        if torch.isnan(value).any() or torch.isinf(value).any():
            raise ValueError("Value包含NaN或Inf值")
        
        q = self.query_proj(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key_proj(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value_proj(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # 安全的注意力权重计算
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # 检查注意力logits
        if torch.isnan(attn_weights).any():
            raise ValueError("注意力logits包含NaN值")
        if torch.isinf(attn_weights).any():
            # 如果有-inf值（来自mask），这是正常的，但+inf不正常
            if torch.isposinf(attn_weights).any():
                raise ValueError("注意力logits包含+Inf值")

        if mask is not None:
            # 确保mask是bool类型，避免PyTorch警告
            if mask.dtype != torch.bool:
                mask = mask.bool()
            attn_weights = attn_weights.masked_fill(~mask, float('-inf'))

        # 安全的softmax计算
        try:
            attn_weights = F.softmax(attn_weights, dim=-1)
        except RuntimeError as e:
            if "CUDA error" in str(e):
                raise RuntimeError(f"CUDA错误在softmax计算中: attn_weights.shape={attn_weights.shape}, "
                                 f"attn_weights.device={attn_weights.device}, "
                                 f"原始错误: {e}")
            else:
                raise
        
        # 检查softmax后的权重
        if torch.isnan(attn_weights).any():
            raise ValueError("Softmax后的注意力权重包含NaN值")
        
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        attn_output = self.out_proj(attn_output)
        
        # 最终输出检查
        if torch.isnan(attn_output).any():
            raise ValueError("最终注意力输出包含NaN值")
        if torch.isinf(attn_output).any():
            raise ValueError("最终注意力输出包含Inf值")
        
        return attn_output


class MultiheadAttentionBlock(nn.Module):
    """
    多头注意力块 (MAB)
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
        # 安全检查：验证输入形状和数值稳定性
        if query.shape != key_value.shape:
            # 如果形状不匹配，只检查最后一维（特征维）是否匹配
            if query.shape[-1] != key_value.shape[-1]:
                raise ValueError(f"Query和key_value的特征维度不匹配: {query.shape[-1]} != {key_value.shape[-1]}")
        
        # 检查输入是否包含NaN或Inf
        if torch.isnan(query).any():
            raise ValueError("Query包含NaN值")
        if torch.isinf(query).any():
            raise ValueError("Query包含Inf值")
        if torch.isnan(key_value).any():
            raise ValueError("Key_value包含NaN值")
        if torch.isinf(key_value).any():
            raise ValueError("Key_value包含Inf值")
        
        attn_output = self.attention(query, key_value, key_value)
        
        # 检查注意力输出
        if torch.isnan(attn_output).any():
            raise ValueError("注意力输出包含NaN值")
        if torch.isinf(attn_output).any():
            raise ValueError("注意力输出包含Inf值")
        
        # 检查形状匹配
        if attn_output.shape != query.shape:
            raise ValueError(f"注意力输出形状与query不匹配: {attn_output.shape} != {query.shape}")
        
        # 安全的残差连接
        try:
            residual_input = query + attn_output
            x = self.norm1(residual_input)
        except RuntimeError as e:
            if "CUDA error" in str(e):
                # 提供更详细的错误信息
                raise RuntimeError(f"CUDA错误在残差连接中: query.shape={query.shape}, "
                                 f"attn_output.shape={attn_output.shape}, "
                                 f"query.device={query.device}, attn_output.device={attn_output.device}, "
                                 f"原始错误: {e}")
            else:
                raise
        
        ff_output = self.feed_forward(x)
        
        # 检查前馈输出
        if torch.isnan(ff_output).any():
            raise ValueError("前馈网络输出包含NaN值")
        if torch.isinf(ff_output).any():
            raise ValueError("前馈网络输出包含Inf值")
        
        # 安全的第二个残差连接
        try:
            x = self.norm2(x + ff_output)
        except RuntimeError as e:
            if "CUDA error" in str(e):
                raise RuntimeError(f"CUDA错误在第二个残差连接中: x.shape={x.shape}, "
                                 f"ff_output.shape={ff_output.shape}, "
                                 f"原始错误: {e}")
            else:
                raise
        
        return x


class ISAB(nn.Module):
    """
    诱导集注意力块 (ISAB)
    修改后：输出K个诱导点在关注了输入集合X后的表示。
    """

    def __init__(self, d_model, num_heads, num_inducing=16, dropout=0.1):
        super(ISAB, self).__init__()
        self.num_inducing = num_inducing
        # 可学习的诱导点，形状 [1, K, D]，1表示批次维度可广播
        self.inducing_points = nn.Parameter(torch.randn(1, num_inducing, d_model))
        # MAB块: 诱导点作为查询，输入集合X作为键和值
        self.mab = MultiheadAttentionBlock(d_model, num_heads, dropout)

    def forward(self, x):
        """
        前向传播

        参数:
            x: 输入张量 (例如, 节点嵌入Enodes)
               - 如果是单个图: [seq_len, d_model]
               - 如果是已填充的密集批次: [batch_size, seq_len, d_model]

        返回:
            h: ISAB输出 (诱导点的表示)
               - 单个图: [num_inducing, d_model]
               - 密集批次: [batch_size, num_inducing, d_model]
        """
        was_unbatched = False
        if x.dim() == 2:  # 输入是 [seq_len, d_model]
            x = x.unsqueeze(0)  # 临时添加批次维度 -> [1, seq_len, d_model]
            was_unbatched = True

        batch_size = x.size(0)

        # 扩展诱导点以匹配批次大小
        # inducing_points_expanded 的形状为 [batch_size, num_inducing, d_model]
        inducing_points_expanded = self.inducing_points.expand(batch_size, -1, -1)

        # MAB: 诱导点作为查询 (query=inducing_points_expanded)，
        #      输入集合x作为键和值 (key_value=x)
        # 输出 h 的形状为 [batch_size, num_inducing, d_model]
        h = self.mab(inducing_points_expanded, x)

        if was_unbatched:
            h = h.squeeze(0)  # 移除假的批次维度 -> [num_inducing, d_model]

        return h


class GlobalPooling(nn.Module):
    """
    全局池化层，使用ISAB（输出诱导点表示）后进行平均池化。
    能够处理单个图、密集批处理的图，以及（通过迭代方式）PyG的稀疏批处理图。
    """

    def __init__(self, d_model, num_heads=8, num_inducing=16, dropout=0.1):
        super(GlobalPooling, self).__init__()
        self.isab_block = ISAB(d_model, num_heads, num_inducing, dropout)
        self.d_model = d_model  # 存储d_model以备错误处理时使用


    def forward(self, x, batch: Optional[torch.Tensor] = None):
        """
        前向传播

        参数:
            x: 节点嵌入
               - PyG批处理: [N_total_nodes, d_model], N_total_nodes是批内所有图的节点总数
               - 密集批处理: [batch_size, seq_len, d_model]
               - 单个图: [seq_len, d_model]
            batch: PyG批处理中的batch向量 [N_total_nodes]，指示每个节点属于哪个图。
                   对于密集批处理或单个图，此参数应为None。

        返回:
            全局嵌入
               - 对于批处理输入: [batch_size, d_model]
               - 对于单个图输入: [d_model]
        """
        if x.dim() == 2 and batch is not None:  # PyG稀疏批处理数据

            # 将稀疏批处理数据按图拆分，对每个图应用ISAB，然后收集结果
            # 注意：这种循环方式对于大量小图的批次效率较低。
            # 更高效的实现可能需要重新设计ISAB以适应稀疏注意力或使用更复杂的批处理技巧。
            
            # 安全地获取batch数量，处理batch为None的情况
            if batch is not None:
                num_graphs = batch.max().item() + 1
            else:
                # 如果没有batch信息，假设只有一个图
                num_graphs = 1
                batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
                
            output_list = []
            for i in range(num_graphs):
                try:
                    graph_nodes_x = x[batch == i]  # 获取当前图的节点 [num_nodes_in_graph_i, d_model]
                    if graph_nodes_x.numel() == 0:  # 处理空图的情况
                        # 可以选择跳过，或者返回一个零向量
                        # 为保持输出形状一致，返回零向量可能更好，但需要注意下游任务
                        output_list.append(torch.zeros(self.d_model, device=x.device, dtype=x.dtype))
                        continue

                    # 确保输入数据的连续性和数值稳定性
                    graph_nodes_x = graph_nodes_x.contiguous()
                    
                    # 检查数值稳定性
                    if torch.isnan(graph_nodes_x).any() or torch.isinf(graph_nodes_x).any():
                        print(f"警告: 图 {i} 包含 NaN 或 Inf 值，使用零向量替代")
                        output_list.append(torch.zeros(self.d_model, device=x.device, dtype=x.dtype))
                        continue
                    
                    # ISAB期望输入 [batch_size (1 for single graph), seq_len, d_model]
                    # isab_block的输出将是 [1, num_inducing, d_model]
                    graph_input = graph_nodes_x.unsqueeze(0).contiguous()
                    inducing_point_repr_per_graph = self.isab_block(graph_input)
                    
                    # 对诱导点表示进行平均池化，得到当前图的全局嵌入 [d_model]
                    pooled_per_graph = torch.mean(inducing_point_repr_per_graph.squeeze(0), dim=0)
                    output_list.append(pooled_per_graph)
                    
                    # 清理中间张量以减少内存碎片
                    del graph_nodes_x, graph_input, inducing_point_repr_per_graph
                    
                except RuntimeError as e:
                    if "CUBLAS" in str(e) or "illegal memory access" in str(e):
                        print(f"CUDA错误在图 {i}，节点数: {(batch == i).sum().item()}，使用零向量替代")
                        output_list.append(torch.zeros(self.d_model, device=x.device, dtype=x.dtype))
                        # 强制清理GPU缓存
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    else:
                        raise e

            if not output_list:  # 如果所有图都是空的
                # 根据批次大小返回一个合适的零张量
                # 如果我们无法从batch确定原始批次大小，这可能会有问题
                # 但通常情况下，至少会有一个非空图
                # 作为后备，如果真的所有图都空，可能需要返回一个特定形状的零张量或报错
                if num_graphs > 0:  # 即使图是空的，batch向量也会指示图的数量
                    return torch.zeros(num_graphs, self.d_model, device=x.device, dtype=x.dtype)
                else:  # 如果连num_graphs都是0（例如输入x为空，batch也为空）
                    return torch.empty(0, self.d_model, device=x.device, dtype=x.dtype)

            global_embedding = torch.stack(output_list, dim=0)  # [num_graphs, d_model]

        else:  # 处理单个图 [seq_len, d_model] 或密集批处理 [batch_size, seq_len, d_model]
            # ISAB的输出将是 [num_inducing, d_model] 或 [batch_size, num_inducing, d_model]
            inducing_point_representations = self.isab_block(x)

            # 沿着诱导点维度进行平均池化
            if inducing_point_representations.dim() == 3:  # 密集批处理 [batch_size, num_inducing, d_model]
                global_embedding = torch.mean(inducing_point_representations, dim=1)  # 平均K维 -> [batch_size, d_model]
            elif inducing_point_representations.dim() == 2:  # 单个图 [num_inducing, d_model]
                global_embedding = torch.mean(inducing_point_representations, dim=0)  # 平均K维 -> [d_model]
            else:
                # 不太可能发生，但作为保护
                global_embedding = inducing_point_representations

        return global_embedding