"""
SUCF: Structurally-Gated Graph-Map Network with Mamba
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_batch

# 导入基础组件
from .esm_projection_head import ESMProjectionHead
from .relational_gvp import RGVPEncoder
from .relational_gatv3 import RGATv3Block
from .fusion_mechanisms import CrossAttention
from .pooling_layers import GlobalPooling
from .amp_multimodal_model import StructuralFeatureProjection, ActivityHead

# 导入SGG-Net特有组件
from .sgg_net_components import (
    LaplacianPositionalEncoding, 
    GRUGate, 
    MambaLayer, 
    PLDDTGating
)

import logging

logger = logging.getLogger(__name__)


class SUCF(nn.Module):
    """
    SGG-Net: 结构引导的图谱网络
    
    三层架构:
    1. pLDDT门控的关系图谱生成
    2. 图谱引导的序列特征精炼  
    3. Mamba驱动的最终融合
    """
    
    def __init__(self, config):
        super(SUCF, self).__init__()

        # 解析配置
        self.config = config
        
        # ESM配置
        esm_config = config.get('esm', {})
        self.esm_model_name = esm_config.get('base_model_name_or_path', 'facebook/esm2_t36_3B_UR50D')
        self.esm_dim = esm_config.get('output_dim', 2560)
        self.esm_repr_layer = esm_config.get('repr_layer', 36)
        
        # 架构配置 - 使用与训练时一致的默认值
        arch_config = config.get('architecture', {})
        self.hidden_dim = arch_config.get('hidden_dim', 512)
        # 关键：这些默认值必须与训练时完全一致
        self.node_scalar_dim = arch_config.get('node_scalar_dim', 22)  # 修正：训练时使用22维
        self.node_vector_dim = arch_config.get('node_vector_dim', 1)
        self.edge_scalar_dim = arch_config.get('edge_scalar_dim', 10)  # 修正：训练时使用10维  
        self.dropout = arch_config.get('dropout', 0.1)
        
        # *** 消融实验控制参数 ***
        self.ablation_type = arch_config.get('ablation_type', 'baseline')
        self.use_plddt_gate = arch_config.get('use_plddt_gate', True)
        self.use_structure_branch = arch_config.get('use_structure_branch', True)
        self.ignore_graph_data = arch_config.get('ignore_graph_data', False)
        self.use_rgat = arch_config.get('use_rgat', True)
        self.use_cross_attention = arch_config.get('use_cross_attention', True)
        self.use_gru_gating = arch_config.get('use_gru_gating', True)
        self.use_simple_fusion = arch_config.get('use_simple_fusion', False)
        self.use_mamba = arch_config.get('use_mamba', True)
        self.use_transformer_replacement = arch_config.get('use_transformer_replacement', False)
        
        # *** 序列引导相关消融参数 ***
        self.use_sequence_guidance = arch_config.get('use_sequence_guidance', True)
        self.use_guided_bias = arch_config.get('use_guided_bias', True)
        self.use_sequence_informed_attention = arch_config.get('use_sequence_informed_attention', True)
        self.use_vanilla_gat = arch_config.get('use_vanilla_gat', False)
        self.use_confidence_weighting = arch_config.get('use_confidence_weighting', True)
        self.use_uncertainty_handling = arch_config.get('use_uncertainty_handling', True)
        self.use_sequence_branch = arch_config.get('use_sequence_branch', True)
        self.use_guided_cross_attention = arch_config.get('use_guided_cross_attention', True)
        
        # *** no_scgg消融实验相关参数 ***
        self.use_scgg_framework = arch_config.get('use_scgg_framework', True)
        self.use_naive_gnn = arch_config.get('use_naive_gnn', False)
        self.use_standard_gat = arch_config.get('use_standard_gat', False)
        self.disable_advanced_features = arch_config.get('disable_advanced_features', False)
        
        # *** 层堆叠配置参数 ***
        self.rgat_layers = arch_config.get('rgat_layers', 3)
        self.gvp_layers = arch_config.get('gvp_layers', 1)
        self.fusion_layers = arch_config.get('fusion_layers', 6)
        self.mamba_layers = arch_config.get('mamba_layers', 2)
        self.transformer_layers = arch_config.get('transformer_layers', 6)
        self.lstm_layers = arch_config.get('lstm_layers', 3)
        
        # 其他架构参数
        self.transformer_heads = arch_config.get('transformer_heads', 8)
        
        logger.info(f"SGG-Net消融配置: {self.ablation_type}")
        if self.ablation_type != 'baseline':
            # 基础消融参数
            logger.info(f"  - use_plddt_gate: {self.use_plddt_gate}")
            logger.info(f"  - use_structure_branch: {self.use_structure_branch}")
            logger.info(f"  - use_gru_gating: {self.use_gru_gating}")
            logger.info(f"  - use_mamba: {self.use_mamba}")
            logger.info(f"  - use_cross_attention: {self.use_cross_attention}")
            
            # 序列引导相关参数
            logger.info(f"  - use_sequence_guidance: {self.use_sequence_guidance}")
            logger.info(f"  - use_guided_bias: {self.use_guided_bias}")
            logger.info(f"  - use_vanilla_gat: {self.use_vanilla_gat}")
            logger.info(f"  - use_sequence_informed_attention: {self.use_sequence_informed_attention}")
            logger.info(f"  - use_guided_cross_attention: {self.use_guided_cross_attention}")
            logger.info(f"  - use_confidence_weighting: {self.use_confidence_weighting}")
            logger.info(f"  - use_uncertainty_handling: {self.use_uncertainty_handling}")
            logger.info(f"  - use_sequence_branch: {self.use_sequence_branch}")
            
            # no_scgg消融实验相关参数
            logger.info(f"  - use_scgg_framework: {self.use_scgg_framework}")
            logger.info(f"  - use_naive_gnn: {self.use_naive_gnn}")
            logger.info(f"  - use_standard_gat: {self.use_standard_gat}")
            logger.info(f"  - disable_advanced_features: {self.disable_advanced_features}")
        
        # SGG-Net特有配置
        self.laplacian_k = arch_config.get('laplacian_k', 8)
        self.rgat_heads = arch_config.get('rgat_heads', 4)
        self.cross_attention_heads = arch_config.get('cross_attention_heads', 8)
        self.mamba_d_state = arch_config.get('mamba_d_state', 16)
        self.mamba_d_conv = arch_config.get('mamba_d_conv', 4)
        self.mamba_expand = arch_config.get('mamba_expand', 2)
        
        # --- 0. 输入编码层 ---
        self._build_input_encoders()
        
        # --- 1. 结构图谱生成层 ---
        self._build_structure_mapper()
        
        # --- 2. 序列精炼层 ---
        self._build_sequence_refiner()
        
        # --- 3. 最终融合层 ---
        self._build_final_fusion()
        
        # --- 4. 预测层 ---
        self._build_prediction_heads()
        
        # 初始化权重
        self._init_weights()
        
        logger.info(f"SGG-Net initialized with hidden_dim={self.hidden_dim}")
        
    def _build_input_encoders(self):
        """构建输入编码器"""
        # ESM序列编码器 - 根据实际的ESMProjectionHead接口调用
        self.esm_projection = ESMProjectionHead(
            in_dim=2560,  # ESM2的标准输出维度
            out_dim=self.hidden_dim
        )
        
        # 结构编码器 - 根据消融参数决定是否构建
        if self.use_structure_branch and not self.disable_advanced_features:
            self.rgvp_encoder = RGVPEncoder(
                node_input_scalar_dim=self.node_scalar_dim,  # 22
                node_input_vector_dim=self.node_vector_dim,  # 1
                edge_input_scalar_dim=self.edge_scalar_dim,  # 10
                output_scalar_dim=128,
                output_vector_dim=16,
                num_layers=self.gvp_layers  # 使用配置的层数
            )
            logger.info(f"  - GVP 层数: {self.gvp_layers}")
            
            # 结构特征投影
            self.struct_projection = StructuralFeatureProjection(
                scalar_dim=128,
                vector_dim=16,
                output_dim=self.hidden_dim
            )
            
            # 拉普拉斯位置编码
            self.pos_encoding = LaplacianPositionalEncoding(k=self.laplacian_k)
            self.pos_enc_linear = nn.Linear(self.laplacian_k, self.hidden_dim)
        else:
            # 消融：不创建结构编码器，节省参数
            logger.info("  - 消融：不创建结构编码器，使用序列特征替代")
            self.rgvp_encoder = None
            self.struct_projection = None
            self.pos_encoding = None
            self.pos_enc_linear = None
        
    def _build_structure_mapper(self):
        """构建结构图谱生成器（根据消融类型条件性构建）"""
        if not self.use_structure_branch:
            # 完全不使用结构分支
            self.structure_mapper = nn.Identity()
            logger.info("  - 消融：完全移除结构分支")
            return
            
        if not self.use_scgg_framework:
            # no_scgg消融：完全移除GAT/RGAT，直接使用GVP输出
            self.structure_mapper = nn.Identity()
            logger.info("  - no_scgg消融：移除所有GAT/RGAT，直接使用GVP输出作为结构图谱")
            
        else:
            # 正常的SCGG框架：使用关系型图注意力网络
            self.structure_mapper = nn.ModuleList()
            
            for layer_idx in range(self.rgat_layers):
                # 根据消融参数决定是否使用序列引导偏置
                seq_bias_dim = None
                if (self.use_sequence_guidance and 
                    self.use_guided_bias and 
                    not self.use_vanilla_gat):
                    seq_bias_dim = self.hidden_dim  # 使用序列特征作为偏置
                
                rgat_layer = RGATv3Block(
                    in_channels=self.hidden_dim,
                    out_channels=self.hidden_dim // self.rgat_heads,  # 每个头的输出维度
                    heads=self.rgat_heads,
                    dropout=self.dropout,
                    edge_dim=self.edge_scalar_dim,  # 使用边特征维度
                    seq_bias_dim=seq_bias_dim  # 序列引导偏置维度
                )
                self.structure_mapper.append(rgat_layer)
            
            logger.info(f"  - SCGG框架：RGAT层数: {self.rgat_layers}")
            logger.info(f"    - use_sequence_guidance: {self.use_sequence_guidance}")
            logger.info(f"    - use_guided_bias: {self.use_guided_bias}")
            logger.info(f"    - use_vanilla_gat: {self.use_vanilla_gat}")
            if seq_bias_dim:
                logger.info(f"    - 序列引导偏置维度: {seq_bias_dim}")
        
        # pLDDT门控 - 消融实验时完全移除，不创建任何组件
        if self.use_plddt_gate:
            # pLDDT置信度门控
            self.plddt_gating = PLDDTGating(self.hidden_dim)
            logger.info("  - 启用pLDDT门控")
        else:
            logger.info("  - 消融：移除pLDDT门控")
        
    def _build_sequence_refiner(self):
        """构建序列精炼器（根据消融类型条件性构建）"""
        if self.use_cross_attention:
            # 交叉注意力机制 (结构->序列) - 支持融合层数配置
            self.seq_refiner = CrossAttention(
                hidden_dim=self.hidden_dim,
                num_heads=self.cross_attention_heads,
                dropout=self.dropout
            )
            logger.info(f"  - Cross-Attention 启用，头数: {self.cross_attention_heads}")
        else:
            # 消融：不使用交叉注意力
            self.seq_refiner = nn.Identity()
            logger.info("  - 消融：移除交叉注意力")
        
        if self.use_gru_gating:
            # GRU门控更新
            self.seq_gate = GRUGate(
                state_dim=self.hidden_dim,
                input_dim=self.hidden_dim
            )
            logger.info("  - GRU门控 启用")
        else:
            # 消融：使用简单融合替代GRU门控
            if self.use_simple_fusion:
                self.seq_gate = nn.Sequential(
                    nn.Linear(self.hidden_dim * 2, self.hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(self.dropout)
                )
                logger.info("  - 消融：使用简单融合替代GRU门控")
            else:
                self.seq_gate = nn.Identity()
                logger.info("  - 消融：移除GRU门控")
        
    def _build_final_fusion(self):
        """构建最终融合层（根据消融类型条件性构建）"""
        # 反向交叉注意力 (序列->结构)
        if self.use_cross_attention:
            self.struct_checker = CrossAttention(
                hidden_dim=self.hidden_dim,
                num_heads=self.cross_attention_heads,
                dropout=self.dropout
            )
        else:
            self.struct_checker = nn.Identity()
        
        if self.use_mamba and not self.disable_advanced_features:
            # Mamba最终整合层 - 支持多层堆叠
            self.final_fusion_layers = nn.ModuleList()
            for i in range(self.mamba_layers):
                self.final_fusion_layers.append(
                    MambaLayer(
                        d_model=self.hidden_dim * 3,  # 拼接3个特征流
                        d_state=self.mamba_d_state,
                        d_conv=self.mamba_d_conv,
                        expand=self.mamba_expand
                    )
                )
            logger.info(f"  - Mamba 层数: {self.mamba_layers}")
        elif self.use_transformer_replacement:
            # 消融：使用Transformer替代Mamba - 支持多层配置
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.hidden_dim * 3,
                nhead=self.transformer_heads,
                dropout=self.dropout,
                batch_first=True
            )
            self.final_fusion_layers = nn.TransformerEncoder(
                encoder_layer, 
                num_layers=self.transformer_layers
            )
            logger.info(f"  - Transformer 层数: {self.transformer_layers}")
        else:
            # 消融：不使用Mamba，使用更简单的MLP
            if self.disable_advanced_features:
                # no_scgg消融：使用非常简单的融合
                self.final_fusion_layers = nn.Sequential(
                    nn.Linear(self.hidden_dim * 3, self.hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(self.dropout)
                )
                logger.info("  - 消融：使用简化MLP替代Mamba")
            else:
                # 标准消融：使用稍复杂的MLP
                self.final_fusion_layers = nn.Sequential(
                    nn.Linear(self.hidden_dim * 3, self.hidden_dim * 2),
                    nn.ReLU(),
                    nn.Dropout(self.dropout),
                    nn.Linear(self.hidden_dim * 2, self.hidden_dim * 3),
                    nn.ReLU(),
                    nn.Dropout(self.dropout)
                )
                logger.info("  - 消融：使用MLP替代Mamba")
        
        # 投影回标准维度
        self.final_projection = nn.Sequential(
            nn.Linear(self.hidden_dim * 3, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.Dropout(self.dropout)
        )
        
    def _build_prediction_heads(self):
        """构建预测头"""
        # 全局池化
        self.global_pooling = GlobalPooling(
            d_model=self.hidden_dim,
            num_heads=8,
            num_inducing=16,
            dropout=self.dropout
        )
        
        # 活性预测头
        self.activity_predictor = ActivityHead(
            input_dim=self.hidden_dim,
            hidden_dim=256,
            output_dim=1,
            dropout=self.dropout * 1.5  # 预测头使用更高的dropout
        )
        
    def _init_weights(self):
        """初始化权重"""
        def init_linear_layers(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.1)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        
        # 只初始化新添加的层，保持预训练模型权重
        if hasattr(self, 'pos_enc_linear') and self.pos_enc_linear is not None:
            self.pos_enc_linear.apply(init_linear_layers)
        if hasattr(self, 'final_projection') and self.final_projection is not None:
            self.final_projection.apply(init_linear_layers)
    
    def forward(self, data):
        """
        SGG-Net前向传播
        
        Args:
            data: PyG数据对象，包含序列和结构信息
            
        Returns:
            output_dict: 包含预测结果和中间特征的字典
        """
        # --- 0. 输入编码 ---
        # 序列编码 - 从data.amp_embedding获取ESM特征
        if hasattr(data, 'amp_embedding') and data.amp_embedding is not None:
            seq_emb_0 = self.esm_projection(data.amp_embedding)  # [N, hidden_dim]
        else:
            # 如果没有ESM嵌入，使用零向量占位
            num_nodes = data.x.size(0)
            seq_emb_0 = torch.zeros(num_nodes, self.hidden_dim, device=data.x.device)
        
        # 结构编码（根据消融类型决定是否使用）
        if (self.use_structure_branch and not self.ignore_graph_data and 
            hasattr(self, 'rgvp_encoder') and self.rgvp_encoder is not None):
            # 直接使用node_vector作为节点向量特征
            struct_scalar, struct_vector = self.rgvp_encoder(
                x_s_in=data.x,  # 节点标量特征 [N, 22]
                x_v_in=data.node_vector,  # 节点向量特征 [N, 1, 3]
                edge_index=data.edge_index,
                edge_attr=data.edge_attr,  # 边标量特征 [E, 10]
                edge_vector=data.edge_vector  # 边向量特征 [E, 1, 3]
            )
            struct_emb_0 = self.struct_projection(struct_scalar, struct_vector)  # [N, hidden_dim]
            
            # 添加位置编码
            if hasattr(self, 'pos_encoding') and self.pos_encoding is not None:
                pos_enc = self.pos_encoding(data)  # [N, laplacian_k]
                pos_emb = self.pos_enc_linear(pos_enc)  # [N, hidden_dim]
                struct_emb_0 = struct_emb_0 + pos_emb
        else:
            # 消融：不使用结构分支，用序列特征替代
            struct_emb_0 = seq_emb_0.clone()
            logger.debug("消融：跳过结构编码，使用序列特征替代")
        
        # --- 第1层: pLDDT门控的关系图谱生成 ---
        if self.use_scgg_framework and self.use_structure_branch and not self.ignore_graph_data:
            # 正常SCGG框架：通过多层RGAT生成原始结构图谱
            raw_structure_map = struct_emb_0
            
            # 如果是多层RGAT（ModuleList），逐层处理
            if isinstance(self.structure_mapper, nn.ModuleList):
                for layer_idx, rgat_layer in enumerate(self.structure_mapper):
                    # SCGG框架：使用RGATv3Block
                    seq_features = None
                    if (self.use_sequence_guidance and 
                        self.use_guided_bias and 
                        not self.use_vanilla_gat):
                        seq_features = seq_emb_0  # 使用序列特征作为引导偏置
                    
                    raw_structure_map = rgat_layer(
                        x=raw_structure_map,
                        edge_index=data.edge_index,
                        edge_attr=data.edge_attr,
                        seq_features=seq_features  # 传递序列特征作为引导偏置
                    )
                    logger.debug(f"SCGG RGAT Layer {layer_idx + 1} 输出形状: {raw_structure_map.shape}")
            else:
                # 单层或恒等映射
                raw_structure_map = self.structure_mapper(struct_emb_0)
                logger.debug("使用恒等映射或单层RGAT")
                
        elif not self.use_scgg_framework and self.use_structure_branch and not self.ignore_graph_data:
            # no_scgg消融：直接使用GVP输出，跳过所有GAT/RGAT处理
            raw_structure_map = struct_emb_0
            logger.debug("no_scgg消融：直接使用GVP输出作为结构图谱，跳过GAT/RGAT")
            
        else:
            # 其他消融：不使用结构分支，直接使用结构特征
            raw_structure_map = struct_emb_0
            logger.debug("消融：跳过结构图谱生成，直接使用结构特征")
        
        # 应用pLDDT门控
        if self.use_plddt_gate and hasattr(self, 'plddt_gating') and hasattr(data, 'plddt') and data.plddt is not None:
            structure_map = self.plddt_gating(raw_structure_map, data.plddt)
            logger.debug("应用pLDDT门控")
        else:
            # 消融：不使用pLDDT门控，或没有pLDDT信息
            # no_scgg消融：直接使用GVP输出作为最终的结构图谱
            structure_map = raw_structure_map
            if not self.use_plddt_gate:
                logger.debug("消融：跳过pLDDT门控，直接使用原始结构图谱")
            elif not hasattr(data, 'plddt') or data.plddt is None:
                logger.debug("无pLDDT信息，跳过门控")
        
        # --- 第2层: 图谱引导的序列特征精炼 ---
        if self.use_cross_attention and self.use_guided_cross_attention:
            # 结构图谱作为Query，序列特征作为Key/Value
            refined_seq_features = self.seq_refiner(
                query=structure_map,
                key_value=seq_emb_0
            )  # [N, hidden_dim]
        else:
            # 消融：不使用交叉注意力或引导交叉注意力
            refined_seq_features = seq_emb_0
            if not self.use_cross_attention:
                logger.debug("消融：跳过交叉注意力精炼")
            elif not self.use_guided_cross_attention:
                logger.debug("消融：跳过引导交叉注意力")
        
        # GRU门控更新序列特征
        if self.use_gru_gating:
            seq_emb_1 = self.seq_gate(
                state=seq_emb_0,
                input_features=refined_seq_features
            )  # [N, hidden_dim]
        elif self.use_simple_fusion:
            # 消融：使用简单融合
            combined = torch.cat([seq_emb_0, refined_seq_features], dim=-1)
            seq_emb_1 = self.seq_gate(combined)
            logger.debug("消融：使用简单融合替代GRU门控")
        else:
            # 消融：不使用门控，直接使用精炼特征
            seq_emb_1 = refined_seq_features
            logger.debug("消融：跳过GRU门控")
        
        # --- 第3层: Mamba驱动的最终融合 ---
        # 反向交叉注意力 (序列->结构检查)
        if self.use_cross_attention and self.use_guided_cross_attention:
            checked_struct_features = self.struct_checker(
                query=seq_emb_1,
                key_value=struct_emb_0
            )  # [N, hidden_dim]
        else:
            # 消融：不使用反向交叉注意力或引导交叉注意力
            checked_struct_features = struct_emb_0
            if not self.use_cross_attention:
                logger.debug("消融：跳过反向交叉注意力")
            elif not self.use_guided_cross_attention:
                logger.debug("消融：跳过引导反向交叉注意力")
        
        # 拼接三个特征流
        logger.debug(f"特征拼接前维度检查:")
        logger.debug(f"  - struct_emb_0: {struct_emb_0.shape}")
        logger.debug(f"  - seq_emb_1: {seq_emb_1.shape}")
        logger.debug(f"  - checked_struct_features: {checked_struct_features.shape}")
        
        combined_features = torch.cat([
            struct_emb_0,           # 原始结构特征
            seq_emb_1,              # 精炼后的序列特征  
            checked_struct_features  # 检查后的结构特征
        ], dim=-1)  # [N, hidden_dim * 3]
        
        logger.debug(f"拼接后combined_features维度: {combined_features.shape}")
        logger.debug(f"期望维度: [N, {self.hidden_dim * 3}]")
        
        # 通过Mamba或替代方案进行最终整合
        if self.use_mamba and not self.disable_advanced_features:
            # 使用多层Mamba进行融合
            current_features = combined_features
            for i, mamba_layer in enumerate(self.final_fusion_layers):
                current_features = mamba_layer(
                    current_features, 
                    batch=data.batch
                )  # [N, hidden_dim * 3]
                logger.debug(f"通过Mamba层{i+1}/{len(self.final_fusion_layers)}")
            fused_features_wide = current_features
        elif self.use_transformer_replacement:
            # 消融：使用Transformer替代Mamba
            # 需要处理batch维度
            dense_features, mask = to_dense_batch(combined_features, data.batch)  # [B, max_N, hidden_dim*3]
            transformer_out = self.final_fusion_layers(dense_features)  # [B, max_N, hidden_dim*3]
            # 转回稀疏格式
            fused_features_wide = transformer_out[mask]  # [N, hidden_dim * 3]
            logger.debug(f"消融：使用{self.transformer_layers}层Transformer替代Mamba")
        else:
            # 消融：简单融合层
            fused_features_wide = self.final_fusion_layers(combined_features)
            if self.disable_advanced_features:
                logger.debug("消融：使用简化MLP替代Mamba")
            else:
                logger.debug("消融：使用标准MLP替代Mamba")
        
        # 投影回标准维度
        fused_node_embedding = self.final_projection(fused_features_wide)  # [N, hidden_dim]
        
        # --- 4. 预测 ---
        # 全局池化
        global_embedding = self.global_pooling(fused_node_embedding, data.batch)  # [B, hidden_dim]
        
        # 活性预测
        activity_pred = self.activity_predictor(global_embedding)  # [B, 1]
        
        # --- 构造输出字典 ---
        output_dict = {
            # 主要预测结果
            'activity_pred': activity_pred,
            
            # 中间特征用于损失计算
            'seq_global': self.global_pooling(seq_emb_1, data.batch),
            'struct_global': self.global_pooling(struct_emb_0, data.batch),
            'combined_global': global_embedding,
            
            # 节点级特征用于可视化分析
            'seq_node_features': seq_emb_1,
            'struct_node_features': struct_emb_0,
            'fused_node_features': fused_node_embedding,
            
            # 注意力权重等中间结果
            'structure_map': structure_map,
            'refined_seq_features': refined_seq_features,
            'checked_struct_features': checked_struct_features
        }
        
        return output_dict
    
    def get_model_info(self):
        """获取模型信息"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'model_type': 'SGG-Net',
            'hidden_dim': self.hidden_dim,
            'esm_model': self.esm_model_name
        }


def create_sucf_model(config):
    """
    工厂函数：创建SGG-Net模型
    
    Args:
        config: 模型配置字典
        
    Returns:
        model: SUCF模型实例
    """
    model = SUCF(config)
    
    # 打印模型信息
    model_info = model.get_model_info()
    logger.info(f"Created SUCF model:")
    logger.info(f"  Total parameters: {model_info['total_params']:,}")
    logger.info(f"  Trainable parameters: {model_info['trainable_params']:,}")
    logger.info(f"  Hidden dimension: {model_info['hidden_dim']}")
    
    return model


if __name__ == "__main__":
    pass

