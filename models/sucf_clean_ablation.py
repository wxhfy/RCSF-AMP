"""
Clean Ablation Framework for SUCF

Design Principles:
1. Each ablation only toggles ONE factor
2. Structure remains usable unless specifically ablating structure
3. Fusion path remains consistent across ablations

Control Variables:
- use_structure: True/False - completely disable structure branch
- use_scgc: True/False - disable sequence-guided structural calibration
- use_seq_guide: True/False - disable sequence bias in RGAT
- use_plddt_gate: True/False - disable pLDDT-based gating
- use_sgfn: True/False - use concat instead of SGFN cross-attention
- use_bimamba: True/False - disable Bi-Mamba context modeling
- fusion_type: 'sgfn' / 'concat' - how to fuse the three streams
- calibration_order: 'calibrate_then_fuse' / 'fuse_then_calibrate'
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_batch

from .esm_projection_head import ESMProjectionHead
from .relational_gvp import RGVPEncoder
from .relational_gatv3 import RGATv3Block
from .fusion_mechanisms import CrossAttention
from .pooling_layers import GlobalPooling
from .amp_multimodal_model import StructuralFeatureProjection, ActivityHead
from .sucf_components import (
    LaplacianPositionalEncoding,
    GRUGate,
    MambaLayer,
    PLDDTGating,
    SimpleConfGatedFusion
)


class SUCFCleanAblation(nn.Module):
    """
    SUCF model with clean ablation controls.
    Each dimension can be toggled independently without affecting others.
    """

    def __init__(self, config,
                 use_structure=True,
                 use_scgc=True,
                 use_seq_guide=True,
                 use_plddt_gate=True,
                 use_sgfn=True,
                 use_bimamba=True,
                 fusion_type='sgfn',  # 'sgfn' or 'concat'
                 calibration_order='calibrate_then_fuse',  # 'calibrate_then_fuse' or 'fuse_then_calibrate'
                 # Ablation baselines (P0-2)
                 use_plddt_node_feature=False,  # Baseline 1: pLDDT as additive feature before RGAT
                 use_simple_conf_gated_fusion=False,  # Baseline 2: direct conf-weighted gating (no cross-attention)
                 use_plddt_attention_bias=False,  # Baseline 3: pLDDT bias in RGAT attention
                 ):
        super().__init__()
        self.config = config

        # Clean ablation flags - each controls ONE aspect
        self.use_structure = use_structure
        self.use_scgc = use_scgc
        self.use_seq_guide = use_seq_guide
        self.use_plddt_gate = use_plddt_gate
        self.use_sgfn = use_sgfn
        self.use_bimamba = use_bimamba
        self.fusion_type = fusion_type
        self.calibration_order = calibration_order
        # Baseline flags
        self.use_plddt_node_feature = use_plddt_node_feature
        self.use_simple_conf_gated_fusion = use_simple_conf_gated_fusion
        self.use_plddt_attention_bias = use_plddt_attention_bias

        # Architecture config
        arch_config = config.get('architecture', {})
        self.hidden_dim = arch_config.get('hidden_dim', 512)
        self.node_scalar_dim = arch_config.get('node_scalar_dim', 22)
        self.node_vector_dim = arch_config.get('node_vector_dim', 1)
        self.edge_scalar_dim = arch_config.get('edge_scalar_dim', 10)
        self.dropout = arch_config.get('dropout', 0.1)
        self.rgat_layers = arch_config.get('rgat_layers', 3)
        self.gvp_layers = arch_config.get('gvp_layers', 1)
        self.laplacian_k = arch_config.get('laplacian_k', 8)
        self.rgat_heads = arch_config.get('rgat_heads', 4)
        self.cross_attention_heads = arch_config.get('cross_attention_heads', 8)
        self.mamba_d_state = arch_config.get('mamba_d_state', 16)
        self.mamba_d_conv = arch_config.get('mamba_d_conv', 4)
        self.mamba_expand = arch_config.get('mamba_expand', 2)

        # Sequence input specs
        self.sequence_input_specs = self._prepare_sequence_input_specs()
        self.sequence_feature_names = [spec['attr'] for spec in self.sequence_input_specs]
        self.sequence_combined_dim = sum(spec['in_dim'] for spec in self.sequence_input_specs)

        self._build_input_encoders()

        # Structure branch - only build if using structure
        if self.use_structure:
            self._build_structure_mapper()
        else:
            self.rgvp_encoder = None
            self.struct_projection = None
            self.pos_encoding = None
            self.pos_enc_linear = None
            self.structure_mapper = None
            self.plddt_gating = None

        # Sequence refiner (SCGC or Baseline 2)
        if self.use_scgc and self.use_structure:
            self._build_sequence_refiner()
        elif self.use_simple_conf_gated_fusion and self.use_structure:
            self._build_sequence_refiner()
        else:
            self.seq_refiner = None
            self.seq_gate = None
            self.simple_conf_gate = None

        # Final fusion - always built
        self._build_final_fusion()

        # Prediction heads - always built
        self._build_prediction_heads()
        self._init_weights()

    def _prepare_sequence_input_specs(self):
        sequence_cfg = self.config.get('sequence_inputs', {}) or {}
        specs = []
        if sequence_cfg:
            for key, cfg in sequence_cfg.items():
                attr_name = cfg.get('attr', key)
                in_dim = cfg.get('in_dim')
                if in_dim is None:
                    raise ValueError(f"Sequence input '{key}' missing 'in_dim'")
                specs.append({'name': key, 'attr': attr_name, 'in_dim': int(in_dim)})
        if not specs:
            esm_cfg = self.config.get('esm', {})
            specs.append({
                'name': 'esm', 'attr': 'amp_embedding',
                'in_dim': int(esm_cfg.get('output_dim', 2560))
            })
        return specs

    def _build_input_encoders(self):
        self.sequence_projection = ESMProjectionHead(
            in_dim=self.sequence_combined_dim, out_dim=self.hidden_dim
        )

        # Structure encoder - only if using structure
        if self.use_structure:
            self.rgvp_encoder = RGVPEncoder(
                node_input_scalar_dim=self.node_scalar_dim,
                node_input_vector_dim=self.node_vector_dim,
                edge_input_scalar_dim=self.edge_scalar_dim,
                output_scalar_dim=128,
                output_vector_dim=16,
                num_layers=self.gvp_layers
            )
            self.struct_projection = StructuralFeatureProjection(
                scalar_dim=128, vector_dim=16, output_dim=self.hidden_dim
            )
            self.pos_encoding = LaplacianPositionalEncoding(k=self.laplacian_k)
            self.pos_enc_linear = nn.Linear(self.laplacian_k, self.hidden_dim)

    def _build_structure_mapper(self):
        """Build RGAT layers for structure processing."""
        self.structure_mapper = nn.ModuleList()

        # Sequence bias dimension
        seq_bias_dim = self.hidden_dim if (self.use_seq_guide and self.use_structure) else 0
        # pLDDT attention bias dimension (Baseline 3)
        plddt_bias_dim = self.hidden_dim if (self.use_plddt_attention_bias and self.use_structure) else 0

        for _ in range(self.rgat_layers):
            rgat_layer = RGATv3Block(
                in_channels=self.hidden_dim,
                out_channels=self.hidden_dim // self.rgat_heads,
                heads=self.rgat_heads,
                dropout=self.dropout,
                edge_dim=self.edge_scalar_dim,
                seq_bias_dim=seq_bias_dim,
                plddt_bias_dim=plddt_bias_dim
            )
            self.structure_mapper.append(rgat_layer)

        # pLDDT gating - only if using pLDDT gate
        if self.use_plddt_gate:
            self.plddt_gating = PLDDTGating(self.hidden_dim)
        else:
            self.plddt_gating = None

        # pLDDT node feature projection (Baseline 1: additive injection before RGAT)
        if self.use_plddt_node_feature:
            self.plddt_node_proj = nn.Linear(1, self.hidden_dim)
        else:
            self.plddt_node_proj = None

        # pLDDT projection for attention bias (Baseline 3: project raw pLDDT to hidden_dim)
        if self.use_plddt_attention_bias:
            self.plddt_proj = nn.Linear(1, self.hidden_dim)
        else:
            self.plddt_proj = None

    def _build_sequence_refiner(self):
        """Build SCGC sequence refiner or Baseline 2 alternative."""
        if self.use_simple_conf_gated_fusion and self.use_structure:
            # Baseline 2: direct pLDDT-weighted fusion (no cross-attention)
            self.simple_conf_gate = SimpleConfGatedFusion(self.hidden_dim)
            self.seq_refiner = None
            self.seq_gate = None
        elif self.use_scgc and self.use_structure:
            # Standard SCGC: cross-attention + GRU gate
            self.seq_refiner = CrossAttention(
                hidden_dim=self.hidden_dim,
                num_heads=self.cross_attention_heads,
                dropout=self.dropout
            )
            self.seq_gate = GRUGate(
                state_dim=self.hidden_dim,
                input_dim=self.hidden_dim
            )
            self.simple_conf_gate = None
        else:
            self.seq_refiner = None
            self.seq_gate = None
            self.simple_conf_gate = None

    def _build_final_fusion(self):
        """Build final fusion layer (struct_checker + Bi-Mamba or alternatives)."""
        self.struct_checker = CrossAttention(
            hidden_dim=self.hidden_dim,
            num_heads=self.cross_attention_heads,
            dropout=self.dropout
        )

        if self.use_bimamba:
            self.mamba_layer = MambaLayer(
                d_model=self.hidden_dim * 3,
                d_state=self.mamba_d_state,
                d_conv=self.mamba_d_conv,
                expand=self.mamba_expand
            )
        else:
            self.mamba_layer = None

        self.final_projection = nn.Sequential(
            nn.Linear(self.hidden_dim * 3, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.Dropout(self.dropout)
        )

    def _build_prediction_heads(self):
        self.global_pooling = GlobalPooling(
            d_model=self.hidden_dim,
            num_heads=8,
            num_inducing=16,
            dropout=self.dropout
        )
        self.activity_predictor = ActivityHead(
            input_dim=self.hidden_dim,
            hidden_dim=256,
            output_dim=1,
            dropout=self.dropout * 1.5
        )

    def _init_weights(self):
        def init_linear(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.1)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        if hasattr(self, 'pos_enc_linear') and self.pos_enc_linear is not None:
            self.pos_enc_linear.apply(init_linear)
        if hasattr(self, 'final_projection'):
            self.final_projection.apply(init_linear)

    def forward(self, data):
        batch_index = getattr(data, 'batch', None)

        # === 1. Sequence Encoding ===
        sequence_feature_chunks = []
        for feature_name in self.sequence_feature_names:
            feat = getattr(data, feature_name)
            if feat.dim() == 3 and feat.size(0) == 1:
                feat = feat.squeeze(0)
            sequence_feature_chunks.append(feat)
        combined_seq = torch.cat(sequence_feature_chunks, dim=-1)
        seq_emb = self.sequence_projection(combined_seq)

        # === 2. Structure Encoding ===
        if self.use_structure:
            struct_scalar, struct_vector = self.rgvp_encoder(
                x_s_in=data.x,
                x_v_in=data.node_vector,
                edge_index=data.edge_index,
                edge_attr=data.edge_attr,
                edge_vector=data.edge_vector
            )
            struct_emb = self.struct_projection(struct_scalar, struct_vector)
            pos_enc = self.pos_encoding(data)
            pos_emb = self.pos_enc_linear(pos_enc)
            struct_emb = struct_emb + pos_emb
        else:
            struct_emb = torch.zeros_like(seq_emb)

        # === 3. Structure Processing (RGAT) ===
        if self.use_structure:
            raw_struct_map = struct_emb

            # Baseline 1: pLDDT node feature injection before RGAT
            if self.use_plddt_node_feature and self.plddt_node_proj is not None:
                plddt_feat = self.plddt_node_proj(data.plddt.unsqueeze(-1))
                raw_struct_map = raw_struct_map + plddt_feat

            # Baseline 3: project pLDDT for attention bias
            if self.use_plddt_attention_bias and self.plddt_proj is not None:
                plddt_proj = self.plddt_proj(data.plddt.unsqueeze(-1))
            else:
                plddt_proj = data.plddt if self.use_plddt_attention_bias else None

            for rgat_layer in self.structure_mapper:
                raw_struct_map = rgat_layer(
                    x=raw_struct_map,
                    edge_index=data.edge_index,
                    edge_attr=data.edge_attr,
                    seq_features=seq_emb if self.use_seq_guide else None,
                    plddt_features=plddt_proj
                )
        else:
            raw_struct_map = torch.zeros_like(seq_emb)

        # === 4. pLDDT Gating ===
        if self.use_structure and self.use_plddt_gate and self.plddt_gating is not None:
            struct_map = self.plddt_gating(
                struct_feats=raw_struct_map,
                seq_feats=seq_emb,
                plddt=data.plddt
            )
        else:
            struct_map = raw_struct_map if self.use_structure else torch.zeros_like(seq_emb)

        # === 5. Sequence Refinement (SCGC or Baseline 2) ===
        if self.use_structure:
            if self.use_scgc and self.seq_refiner is not None:
                # Standard SCGC: structure attends to sequence via cross-attention
                refined_seq = self.seq_refiner(
                    query=seq_emb,
                    key_value=struct_map,
                    query_batch=batch_index,
                    key_value_batch=batch_index
                )
                seq_emb_1 = self.seq_gate(state=seq_emb, input_features=refined_seq)
            elif self.use_simple_conf_gated_fusion and self.simple_conf_gate is not None:
                # Baseline 2: direct pLDDT-weighted fusion (no cross-attention)
                seq_emb_1 = self.simple_conf_gate(seq_emb, struct_map, data.plddt)
            else:
                seq_emb_1 = seq_emb
        else:
            seq_emb_1 = seq_emb

        # === 6. Final Fusion ===
        if self.calibration_order == 'calibrate_then_fuse':
            # Standard: calibrate structure then fuse
            if self.use_structure and self.use_sgfn:
                # SGFN: cross-attention refines structure using calibrated sequence as query
                checked_struct = self.struct_checker(
                    query=seq_emb_1,
                    key_value=struct_emb,
                    query_batch=batch_index,
                    key_value_batch=batch_index
                )
            elif self.use_structure:
                # use_sgfn=False -> simple concat baseline: replace SGFN-refined stream
                # with raw structure features (mirrors fuse_then_calibrate concat path)
                checked_struct = struct_emb
            else:
                checked_struct = torch.zeros_like(seq_emb_1)

            combined_features = torch.cat([
                struct_emb if self.use_structure else seq_emb_1,
                seq_emb_1,
                checked_struct
            ], dim=-1)

        else:  # fuse_then_calibrate
            # Alternative: fuse first, then apply structure guidance
            if self.fusion_type == 'concat':
                # Simple concat fusion
                combined_features = torch.cat([
                    struct_emb if self.use_structure else seq_emb_1,
                    seq_emb_1,
                    struct_emb if self.use_structure else seq_emb_1
                ], dim=-1)
            else:
                # SGFN fusion
                if self.use_structure:
                    checked_struct = self.struct_checker(
                        query=seq_emb_1,
                        key_value=struct_emb,
                        query_batch=batch_index,
                        key_value_batch=batch_index
                    )
                else:
                    checked_struct = torch.zeros_like(seq_emb_1)
                combined_features = torch.cat([
                    struct_emb if self.use_structure else seq_emb_1,
                    seq_emb_1,
                    checked_struct
                ], dim=-1)

        # === 7. Context Modeling (Bi-Mamba or alternative) ===
        if self.use_bimamba and self.mamba_layer is not None:
            fused_features = self.mamba_layer(combined_features, batch=data.batch)
        else:
            fused_features = combined_features

        fused_node_embedding = self.final_projection(fused_features)
        global_embedding = self.global_pooling(fused_node_embedding, data.batch)
        activity_pred = self.activity_predictor(global_embedding)

        return {
            'activity_pred': activity_pred,
            'seq_global': self.global_pooling(seq_emb_1, data.batch),
            'struct_global': self.global_pooling(struct_emb, data.batch),
            'combined_global': global_embedding,
            'fused_node_features': fused_node_embedding,
        }


# ============================================================
# Ablation Configuration Presets
# ============================================================

ABLATION_PRESETS = {
    # Matrix A: Clean Component Ablation
    'full': {
        'use_structure': True,
        'use_scgc': True,
        'use_seq_guide': True,
        'use_plddt_gate': True,
        'use_sgfn': True,
        'use_bimamba': True,
        'fusion_type': 'sgfn',
        'calibration_order': 'calibrate_then_fuse',
    },

    'wo_structure': {
        'use_structure': False,
        'use_scgc': False,  # No structure to calibrate
        'use_seq_guide': False,
        'use_plddt_gate': False,
        'use_sgfn': False,  # No structure to fuse
        'use_bimamba': True,
        'fusion_type': 'concat',
        'calibration_order': 'calibrate_then_fuse',
    },

    # KEY: wo_scgc_keep_structure -验证SCGC的真实贡献
    # 只去掉SCGC，保留所有其他模块
    'wo_scgc_keep_structure': {
        'use_structure': True,
        'use_scgc': False,  # ONLY change: no SCGC
        'use_seq_guide': True,  # Keep seq guide in RGAT
        'use_plddt_gate': True,  # Keep pLDDT gate
        'use_sgfn': True,  # Keep SGFN cross-attention
        'use_bimamba': True,  # Keep Bi-Mamba
        'fusion_type': 'sgfn',
        'calibration_order': 'calibrate_then_fuse',
    },

    # KEY: wo_plddt_gate_keep_structure -验证pLDDT gate的真实贡献
    'wo_plddt_gate_keep_structure': {
        'use_structure': True,
        'use_scgc': True,  # Keep SCGC
        'use_seq_guide': True,
        'use_plddt_gate': False,  # ONLY change: no pLDDT gating
        'use_sgfn': True,
        'use_bimamba': True,
        'fusion_type': 'sgfn',
        'calibration_order': 'calibrate_then_fuse',
    },

    # pLDDT gate variants
    'plddt_gate_learned': {
        'use_structure': True,
        'use_scgc': True,
        'use_seq_guide': True,
        'use_plddt_gate': 'learned',  # Learned gate without pLDDT input
        'use_sgfn': True,
        'use_bimamba': True,
        'fusion_type': 'sgfn',
        'calibration_order': 'calibrate_then_fuse',
    },

    'plddt_gate_shuffled': {
        'use_structure': True,
        'use_scgc': True,
        'use_seq_guide': True,
        'use_plddt_gate': 'shuffled',  # Shuffled pLDDT
        'use_sgfn': True,
        'use_bimamba': True,
        'fusion_type': 'sgfn',
        'calibration_order': 'calibrate_then_fuse',
    },

    # wo_seq_guide_keep_structure
    'wo_seq_guide_keep_structure': {
        'use_structure': True,
        'use_scgc': True,
        'use_seq_guide': False,  # ONLY change: no sequence bias
        'use_plddt_gate': True,
        'use_sgfn': True,
        'use_bimamba': True,
        'fusion_type': 'sgfn',
        'calibration_order': 'calibrate_then_fuse',
    },

    # sgfn_concat_keep_structure -验证SGFN是否优于静态拼接
    'sgfn_concat_keep_structure': {
        'use_structure': True,
        'use_scgc': True,
        'use_seq_guide': True,
        'use_plddt_gate': True,
        'use_sgfn': False,  # ONLY change: use concat instead of SGFN
        'use_bimamba': True,
        'fusion_type': 'concat',
        'calibration_order': 'calibrate_then_fuse',
    },

    # wo_bimamba_keep_fusion -验证Bi-Mamba的真实贡献
    'wo_bimamba_keep_fusion': {
        'use_structure': True,
        'use_scgc': True,
        'use_seq_guide': True,
        'use_plddt_gate': True,
        'use_sgfn': True,
        'use_bimamba': False,  # ONLY change: no Bi-Mamba
        'fusion_type': 'sgfn',
        'calibration_order': 'calibrate_then_fuse',
    },

    # Matrix C: Calibration Order Experiments
    'fuse_then_calibrate': {
        'use_structure': True,
        'use_scgc': True,
        'use_seq_guide': True,
        'use_plddt_gate': True,
        'use_sgfn': True,
        'use_bimamba': True,
        'fusion_type': 'sgfn',
        'calibration_order': 'fuse_then_calibrate',
    },

    'direct_fusion_no_calibration': {
        'use_structure': True,
        'use_scgc': False,  # No calibration
        'use_seq_guide': True,
        'use_plddt_gate': False,  # No gating
        'use_sgfn': True,
        'use_bimamba': True,
        'fusion_type': 'sgfn',
        'calibration_order': 'calibrate_then_fuse',
    },

    'plddt_concat_only': {
        'use_structure': True,
        'use_scgc': False,
        'use_seq_guide': False,
        'use_plddt_gate': False,
        'use_sgfn': False,  # Simple concat
        'use_bimamba': True,
        'fusion_type': 'concat',
        'calibration_order': 'calibrate_then_fuse',
    },

    # ============================================================
    # Baseline 1: pLDDT as node feature (additive injection before RGAT)
    # ============================================================
    'plddt_node_feature': {
        'use_structure': True,
        'use_scgc': True,
        'use_seq_guide': True,
        'use_plddt_gate': True,
        'use_sgfn': True,
        'use_bimamba': True,
        'fusion_type': 'sgfn',
        'calibration_order': 'calibrate_then_fuse',
        'use_plddt_node_feature': True,
        'use_simple_conf_gated_fusion': False,
        'use_plddt_attention_bias': False,
    },

    # ============================================================
    # Baseline 2: simple confidence-gated fusion (no cross-attention)
    # ============================================================
    'simple_conf_gated_fusion': {
        'use_structure': True,
        'use_scgc': False,  # Replaced by direct pLDDT-weighted gating
        'use_seq_guide': True,
        'use_plddt_gate': True,
        'use_sgfn': True,
        'use_bimamba': True,
        'fusion_type': 'sgfn',
        'calibration_order': 'calibrate_then_fuse',
        'use_plddt_node_feature': False,
        'use_simple_conf_gated_fusion': True,
        'use_plddt_attention_bias': False,
    },

    # ============================================================
    # Baseline 3: pLDDT attention bias in RGAT (alongside seq_bias)
    # ============================================================
    'plddt_attention_bias': {
        'use_structure': True,
        'use_scgc': True,
        'use_seq_guide': True,
        'use_plddt_gate': True,
        'use_sgfn': True,
        'use_bimamba': True,
        'fusion_type': 'sgfn',
        'calibration_order': 'calibrate_then_fuse',
        'use_plddt_node_feature': False,
        'use_simple_conf_gated_fusion': False,
        'use_plddt_attention_bias': True,
    },
}


def create_clean_ablation_model(config, ablation_name):
    """Factory function to create a clean ablation model."""
    if ablation_name not in ABLATION_PRESETS:
        raise ValueError(f"Unknown ablation: {ablation_name}. Available: {list(ABLATION_PRESETS.keys())}")

    preset = ABLATION_PRESETS[ablation_name]
    return SUCFCleanAblation(config, **preset)