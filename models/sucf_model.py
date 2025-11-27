"""
SUCF: Structurally-Gated Cross-modal Fusion with Mamba
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_batch

# Import foundational components
from .esm_projection_head import ESMProjectionHead
from .relational_gvp import RGVPEncoder
from .relational_gatv3 import RGATv3Block
from .fusion_mechanisms import CrossAttention
from .pooling_layers import GlobalPooling
from .amp_multimodal_model import StructuralFeatureProjection, ActivityHead

# Import SUCF-specific components
from .sucf_components import (
    LaplacianPositionalEncoding, 
    GRUGate, 
    MambaLayer, 
    PLDDTGating
)

import logging

logger = logging.getLogger(__name__)


class SUCF(nn.Module):
    """
    SUCF: Structurally-Gated Cross-modal Fusion Network
    
    Three-stage architecture:
    1. pLDDT-Gated Relational Graph Mapping
    2. Graph-Guided Sequence Feature Refinement
    3. Mamba-Driven Final Fusion
    """
    
    def __init__(self, config):
        super(SUCF, self).__init__()

        # Parse configuration
        self.config = config
        
        # Architecture configuration - use defaults consistent with training
        arch_config = config.get('architecture', {})
        self.hidden_dim = arch_config.get('hidden_dim', 512)
        self.node_scalar_dim = arch_config.get('node_scalar_dim', 22)
        self.node_vector_dim = arch_config.get('node_vector_dim', 1)
        self.edge_scalar_dim = arch_config.get('edge_scalar_dim', 10)
        self.dropout = arch_config.get('dropout', 0.1)
        
        # Layer stacking configuration
        self.rgat_layers = arch_config.get('rgat_layers', 3)
        self.gvp_layers = arch_config.get('gvp_layers', 1)
        self.mamba_layers = arch_config.get('mamba_layers', 2)
        
        # SUCF-specific component configuration
        self.laplacian_k = arch_config.get('laplacian_k', 8)
        self.rgat_heads = arch_config.get('rgat_heads', 4)
        self.cross_attention_heads = arch_config.get('cross_attention_heads', 8)
        self.mamba_d_state = arch_config.get('mamba_d_state', 16)
        self.mamba_d_conv = arch_config.get('mamba_d_conv', 4)
        self.mamba_expand = arch_config.get('mamba_expand', 2)
        
        # Sequence input specification (supports multi-modal concatenation)
        self.sequence_input_specs = self._prepare_sequence_input_specs()
        self.sequence_feature_names = [spec['attr'] for spec in self.sequence_input_specs]
        self.sequence_combined_dim = sum(spec['in_dim'] for spec in self.sequence_input_specs)

        # --- Build model architecture ---
        self._build_input_encoders()
        self._build_structure_mapper()
        self._build_sequence_refiner()
        self._build_final_fusion()
        self._build_prediction_heads()
        
        # Initialize weights
        self._init_weights()
        
        logger.info(f"SUCF model initialized with hidden_dim={self.hidden_dim}")
        
    def _prepare_sequence_input_specs(self):
        """Parses the sequence input configuration into a normalized list."""
        sequence_cfg = self.config.get('sequence_inputs', {}) or {}
        specs = []

        if sequence_cfg:
            for key, cfg in sequence_cfg.items():
                attr_name = cfg.get('attr', key)
                in_dim = cfg.get('in_dim')
                if in_dim is None:
                    raise ValueError(f"Sequence input '{key}' is missing 'in_dim' in the configuration.")
                specs.append({'name': key, 'attr': attr_name, 'in_dim': int(in_dim)})

        if not specs:
            esm_cfg = self.config.get('esm', {})
            specs.append({
                'name': 'esm',
                'attr': 'amp_embedding',
                'in_dim': int(esm_cfg.get('output_dim', 2560))
            })

        return specs

    def _build_input_encoders(self):
        """Builds the input encoders."""
        self.sequence_projection = ESMProjectionHead(
            in_dim=self.sequence_combined_dim,
            out_dim=self.hidden_dim
        )
        
        # Structure encoder
        self.rgvp_encoder = RGVPEncoder(
            node_input_scalar_dim=self.node_scalar_dim,
            node_input_vector_dim=self.node_vector_dim,
            edge_input_scalar_dim=self.edge_scalar_dim,
            output_scalar_dim=128,
            output_vector_dim=16,
            num_layers=self.gvp_layers
        )
        
        # Structure feature projection
        self.struct_projection = StructuralFeatureProjection(
            scalar_dim=128,
            vector_dim=16,
            output_dim=self.hidden_dim
        )
        
        # Laplacian positional encoding
        self.pos_encoding = LaplacianPositionalEncoding(k=self.laplacian_k)
        self.pos_enc_linear = nn.Linear(self.laplacian_k, self.hidden_dim)
        
    def _build_structure_mapper(self):
        """Builds the structure graph mapper."""
        # Relational Graph Attention Network
        self.structure_mapper = nn.ModuleList()
        for _ in range(self.rgat_layers):
            rgat_layer = RGATv3Block(
                in_channels=self.hidden_dim,
                out_channels=self.hidden_dim // self.rgat_heads,
                heads=self.rgat_heads,
                dropout=self.dropout,
                edge_dim=self.edge_scalar_dim,
                seq_bias_dim=self.hidden_dim  # Enable sequence-guided bias
            )
            self.structure_mapper.append(rgat_layer)
        
        # pLDDT confidence gating
        self.plddt_gating = PLDDTGating(self.hidden_dim)
        
    def _build_sequence_refiner(self):
        """Builds the sequence refiner."""
        # Cross-Attention mechanism (Structure -> Sequence)
        self.seq_refiner = CrossAttention(
            hidden_dim=self.hidden_dim,
            num_heads=self.cross_attention_heads,
            dropout=self.dropout
        )
        
        # GRU gate for updates
        self.seq_gate = GRUGate(
            state_dim=self.hidden_dim,
            input_dim=self.hidden_dim
        )
        
    def _build_final_fusion(self):
        """Builds the final fusion layer."""
        # Reverse Cross-Attention (Sequence -> Structure)
        self.struct_checker = CrossAttention(
            hidden_dim=self.hidden_dim,
            num_heads=self.cross_attention_heads,
            dropout=self.dropout
        )
        
        # Mamba for final integration
        self.final_fusion_layers = nn.ModuleList()
        for _ in range(self.mamba_layers):
            self.final_fusion_layers.append(
                MambaLayer(
                    d_model=self.hidden_dim * 3,  # Concatenation of 3 feature streams
                    d_state=self.mamba_d_state,
                    d_conv=self.mamba_d_conv,
                    expand=self.mamba_expand
                )
            )
        
        # Project back to the standard hidden dimension
        self.final_projection = nn.Sequential(
            nn.Linear(self.hidden_dim * 3, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.Dropout(self.dropout)
        )
        
    def _build_prediction_heads(self):
        """Builds the prediction heads."""
        # Global pooling
        self.global_pooling = GlobalPooling(
            d_model=self.hidden_dim,
            num_heads=8,
            num_inducing=16,
            dropout=self.dropout
        )
        
        # Activity prediction head
        self.activity_predictor = ActivityHead(
            input_dim=self.hidden_dim,
            hidden_dim=256,
            output_dim=1,
            dropout=self.dropout * 1.5
        )
        
    def _init_weights(self):
        """Initializes weights."""
        def init_linear_layers(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.1)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        
        self.pos_enc_linear.apply(init_linear_layers)
        self.final_projection.apply(init_linear_layers)
    
    def forward(self, data):
        """
        SUCF forward pass.
        
        Args:
            data: PyG Data object containing sequence and structure information.
            
        Returns:
            output_dict: A dictionary containing prediction results and intermediate features.
        """
        # --- 0. Input Encoding ---
        batch_index = getattr(data, 'batch', None)
        sequence_feature_chunks = []
        for feature_name in self.sequence_feature_names:
            if not hasattr(data, feature_name):
                raise AttributeError(
                    f"Data sample is missing required sequence feature '{feature_name}'."
                )

            feature_tensor = getattr(data, feature_name)
            if feature_tensor.dim() == 3 and feature_tensor.size(0) == 1:
                feature_tensor = feature_tensor.squeeze(0)

            if feature_tensor.size(0) != data.num_nodes:
                raise ValueError(
                    f"Feature '{feature_name}' length {feature_tensor.size(0)} does not match graph nodes {data.num_nodes}."
                )

            sequence_feature_chunks.append(feature_tensor)

        combined_sequence_features = torch.cat(sequence_feature_chunks, dim=-1)
        seq_emb_0 = self.sequence_projection(combined_sequence_features)
        
        struct_scalar, struct_vector = self.rgvp_encoder(
            x_s_in=data.x,
            x_v_in=data.node_vector,
            edge_index=data.edge_index,
            edge_attr=data.edge_attr,
            edge_vector=data.edge_vector
        )
        struct_emb_0 = self.struct_projection(struct_scalar, struct_vector)
        
        pos_enc = self.pos_encoding(data)
        pos_emb = self.pos_enc_linear(pos_enc)
        struct_emb_0 = struct_emb_0 + pos_emb
        
        # --- 1. pLDDT-Gated Relational Graph Mapping ---
        raw_structure_map = struct_emb_0
        for rgat_layer in self.structure_mapper:
            raw_structure_map = rgat_layer(
                x=raw_structure_map,
                edge_index=data.edge_index,
                edge_attr=data.edge_attr,
                seq_features=seq_emb_0  # Pass sequence features for guided bias
            )
        
        structure_map = self.plddt_gating(
            struct_feats=raw_structure_map,
            seq_feats=seq_emb_0,
            plddt=data.plddt
        )
        
        # --- 2. Graph-Guided Sequence Feature Refinement ---
        refined_seq_features = self.seq_refiner(
            query=structure_map,
            key_value=seq_emb_0,
            query_batch=batch_index,
            key_value_batch=batch_index
        )
        
        seq_emb_1 = self.seq_gate(
            state=seq_emb_0,
            input_features=refined_seq_features
        )
        
        # --- 3. Mamba-Driven Final Fusion ---
        checked_struct_features = self.struct_checker(
            query=seq_emb_1,
            key_value=struct_emb_0,
            query_batch=batch_index,
            key_value_batch=batch_index
        )
        
        combined_features = torch.cat([
            struct_emb_0,             # Original structure features
            seq_emb_1,                # Refined sequence features
            checked_struct_features   # Checked structure features
        ], dim=-1)
        
        fused_features = combined_features
        for mamba_layer in self.final_fusion_layers:
            fused_features = mamba_layer(fused_features, batch=data.batch)
        
        fused_node_embedding = self.final_projection(fused_features)
        
        # --- 4. Prediction ---
        global_embedding = self.global_pooling(fused_node_embedding, data.batch)
        activity_pred = self.activity_predictor(global_embedding)
        
        # --- Construct Output Dictionary ---
        return {
            'activity_pred': activity_pred,
            'seq_global': self.global_pooling(seq_emb_1, data.batch),
            'struct_global': self.global_pooling(struct_emb_0, data.batch),
            'combined_global': global_embedding,  
            'fused_node_features': fused_node_embedding,
        }

    def get_model_info(self):
        """Gets model information."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        esm_config = self.config.get('esm', {})
        esm_model_name = esm_config.get('base_model_name_or_path', 'N/A')
        
        return {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'model_type': 'SUCF',
            'hidden_dim': self.hidden_dim,
            'esm_model': esm_model_name
        }


def create_sucf_model(config):
    """
    Factory function: Creates the SUCF model.
    
    Args:
        config: A dictionary containing the model configuration.
        
    Returns:
        model: An instance of the SUCF model.
    """
    model = SUCF(config)
    
    # Log model information
    model_info = model.get_model_info()
    logger.info("Created SUCF model:")
    logger.info(f"  Total parameters: {model_info['total_params']:,}")
    logger.info(f"  Trainable parameters: {model_info['trainable_params']:,}")
    logger.info(f"  Hidden dimension: {model_info['hidden_dim']}")
    
    return model