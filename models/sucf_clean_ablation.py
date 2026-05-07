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
import math
from torch_geometric.utils import to_dense_batch
from torch_scatter import scatter_mean

from .esm_projection_head import ESMProjectionHead
from .relational_gvp import RGVPEncoder
from .relational_gatv3 import RGATv3Block
from .fusion_mechanisms import CrossAttention, SCGCCrossAttention
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
                 # End-to-end confidence pipeline switches (default ON; turned OFF
                 # together with `use_plddt_gate` so the wo_plddt_gate ablation
                 # truly cuts the entire confidence-aware story).
                 use_rgat_conf_gate=True,
                 use_scgc_conf_aware=True,
                 use_sgfn_conf_aware=True,
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
        # Confidence-pipeline switches (gated by `use_plddt_gate` so wo_plddt_gate
        # cuts the entire pipeline as the paper narrative claims).
        self.use_rgat_conf_gate = bool(use_rgat_conf_gate) and bool(use_plddt_gate)
        self.use_scgc_conf_aware = bool(use_scgc_conf_aware) and bool(use_plddt_gate)
        self.use_sgfn_conf_aware = bool(use_sgfn_conf_aware) and bool(use_plddt_gate)

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
                plddt_bias_dim=plddt_bias_dim,
                use_conf_gate=self.use_rgat_conf_gate,
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
            # Standard SCGC: confidence-aware cross-attention + GRU gate.
            # When the confidence pipeline is on, sequence attends to structure
            # with low-pLDDT keys actively suppressed (additive log-bias).
            if self.use_scgc_conf_aware:
                self.seq_refiner = SCGCCrossAttention(
                    hidden_dim=self.hidden_dim,
                    num_heads=self.cross_attention_heads,
                    dropout=self.dropout,
                )
            else:
                self.seq_refiner = CrossAttention(
                    hidden_dim=self.hidden_dim,
                    num_heads=self.cross_attention_heads,
                    dropout=self.dropout,
                )
            self.seq_gate = GRUGate(
                state_dim=self.hidden_dim,
                input_dim=self.hidden_dim
            )
            self.simple_conf_gate = None
            # Round 7: SCGC mid-bin peaked + disagreement trigger.
            # Replaces the always-on calibration with a specialist that fires
            # mainly in the ambiguous-quality region (pLDDT~70) when seq and
            # struct disagree. Outside the mid bin, refined_seq decays toward
            # seq_emb so SCGC behaves as identity in low/high bins.
            self.scgc_mu_mid = nn.Parameter(torch.tensor(0.70))     # plddt-normalised peak
            self.scgc_log_sigma_mid = nn.Parameter(torch.tensor(math.log(0.15)))
            self.scgc_beta = nn.Parameter(torch.tensor(1.0))         # strength scaling
        else:
            self.seq_refiner = None
            self.seq_gate = None
            self.simple_conf_gate = None

    def _build_final_fusion(self):
        """Build final fusion layer (struct_checker + Bi-Mamba or alternatives)."""
        # SGFN structural checker: confidence-aware cross-attention with the
        # *neighbour-averaged* pLDDT signal so its confidence cue is orthogonal
        # to the per-residue pLDDT used by `PLDDTGating`. When the confidence
        # pipeline is off (wo_plddt_gate ablation), fall back to plain attention.
        if self.use_sgfn_conf_aware:
            self.struct_checker = SCGCCrossAttention(
                hidden_dim=self.hidden_dim,
                num_heads=self.cross_attention_heads,
                dropout=self.dropout,
            )
        else:
            self.struct_checker = CrossAttention(
                hidden_dim=self.hidden_dim,
                num_heads=self.cross_attention_heads,
                dropout=self.dropout,
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

        # Reliability projector for Bi-Mamba residual (Fix 5; Round 7 redesign).
        # Round 7 decomposition: reliability = q_prior(monotone) * c_resid(MLP).
        #   q_prior(plddt) = sigmoid(scale * (plddt_norm - bias)) -- enforces
        #     monotonicity in pLDDT so high-quality residues always get more
        #     long-range context, eliminating the round-6 reliability inversion.
        #   c_resid(cos_align, mean_gate, neighbour_conf) ∈ (0, 1) acts as a
        #     bounded compatibility residual that can only modulate, never flip,
        #     the prior.
        # Inputs to c_resid_mlp: [mean_gate_i, cos(seq, struct)_i, neighbour_conf_i]
        self.reliability_q_scale = nn.Parameter(torch.tensor(8.0))
        self.reliability_q_bias = nn.Parameter(torch.tensor(0.7))  # plddt-normalised
        self.reliability_mlp = nn.Sequential(  # role: c_resid for reliability
            nn.Linear(3, 16),
            nn.GELU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )
        # Mixing coefficient between Mamba output and pre-Mamba combined features
        # so the model can fall back to the un-Mambaed concat stream at low
        # reliability without re-using the raw pLDDT signal twice.
        # Round 7: alpha shares the same q_prior to keep both routing paths
        # quality-monotone; the MLP only learns the compatibility residual.
        self.alpha_proj = nn.Sequential(  # role: c_resid for alpha
            nn.Linear(3, 16),
            nn.GELU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

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
        # Round 7: graph-quality 3-expert router for the final fusion.
        # Inputs (per-graph): [mean_plddt_norm, frac_plddt_lt70, var_plddt_norm].
        # Outputs (per-graph): softmax weights over three node-level experts:
        #   * E_seq    — sequence-only stream            (low-quality regime)
        #   * E_mid    — seq + pLDDT-gated structure     (mid-quality regime)
        #   * E_high   — seq + checked struct + mamba    (high-quality regime)
        # The router is small to avoid overfitting on graph-level signals.
        self.quality_router = nn.Sequential(
            nn.Linear(3, 16),
            nn.GELU(),
            nn.Linear(16, 3),
        )
        # Node-level projections for the balanced (E_mid) expert.
        # E_seq reuses seq_emb_1 directly; E_high reuses fused_node_embedding.
        self.expert_balanced_proj = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.Dropout(self.dropout),
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

            # Raw pLDDT passed into RGAT for sender-side confidence gating.
            # Setting to None when use_rgat_conf_gate is False fully cuts the
            # confidence pipeline (used by `wo_plddt_gate` ablation).
            rgat_plddt_raw = data.plddt if self.use_rgat_conf_gate else None

            for rgat_layer in self.structure_mapper:
                raw_struct_map = rgat_layer(
                    x=raw_struct_map,
                    edge_index=data.edge_index,
                    edge_attr=data.edge_attr,
                    seq_features=seq_emb if self.use_seq_guide else None,
                    plddt_features=plddt_proj,
                    plddt_raw=rgat_plddt_raw,
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
            gate_per_residue = self.plddt_gating._last_gate_scalar  # [N, 1]
        else:
            struct_map = raw_struct_map if self.use_structure else torch.zeros_like(seq_emb)
            gate_per_residue = None

        # Pre-compute neighbour-averaged confidence (used by SGFN + reliability MLP).
        if self.use_structure:
            try:
                neighbour_plddt = scatter_mean(
                    data.plddt[data.edge_index[0]],
                    data.edge_index[1],
                    dim=0,
                    dim_size=data.plddt.size(0),
                )
                # Nodes with no incoming edges fall back to their own pLDDT.
                no_neighbour_mask = neighbour_plddt == 0
                if no_neighbour_mask.any():
                    neighbour_plddt = torch.where(no_neighbour_mask, data.plddt, neighbour_plddt)
            except Exception:
                neighbour_plddt = data.plddt
        else:
            neighbour_plddt = None

        # === 5. Sequence Refinement (SCGC or Baseline 2) ===
        # Round 7: SCGC is reformulated as a mid-bin peaked + disagreement-
        # triggered specialist. The cross-attention output (refined_seq) is
        # mixed with the raw seq_emb by a per-residue strength factor that
        # peaks in the ambiguous-quality bin and only fires when seq/struct
        # disagree. Outside the mid bin or under high agreement, SCGC decays
        # to identity, leaving the full's low-bin role to the gate and the
        # high-bin role to Bi-Mamba.
        scgc_strength = None
        if self.use_structure:
            if self.use_scgc and self.seq_refiner is not None:
                # SCGC: sequence query attends to (calibrated) structure key,
                # with low-pLDDT structural keys actively suppressed when
                # confidence-aware variant is enabled.
                if isinstance(self.seq_refiner, SCGCCrossAttention) and self.use_scgc_conf_aware:
                    refined_seq = self.seq_refiner(
                        query=seq_emb,
                        key_value=struct_map,
                        query_batch=batch_index,
                        key_value_batch=batch_index,
                        plddt_key=data.plddt,
                    )
                else:
                    refined_seq = self.seq_refiner(
                        query=seq_emb,
                        key_value=struct_map,
                        query_batch=batch_index,
                        key_value_batch=batch_index,
                    )

                # Round 7: mid-bin peaked + disagreement-triggered strength.
                plddt_norm = (data.plddt / 100.0).clamp(0.0, 1.0)            # [N]
                sigma_mid = self.scgc_log_sigma_mid.exp().clamp(min=0.05, max=0.5)
                g_mid = torch.exp(-((plddt_norm - self.scgc_mu_mid) ** 2) / (2.0 * sigma_mid ** 2))  # [N]
                # Detach the trigger so SCGC cannot game its own activation.
                cos_sd = F.cosine_similarity(seq_emb, struct_map, dim=-1).clamp(-1.0, 1.0).detach()
                disagreement = (1.0 - cos_sd).clamp(0.0, 2.0)                # [N]
                # sigmoid(beta) keeps strength bounded; init beta=1 ⇒ ~0.73.
                scgc_strength = (g_mid * disagreement * torch.sigmoid(self.scgc_beta)).unsqueeze(-1)  # [N, 1]
                refined_seq = scgc_strength * refined_seq + (1.0 - scgc_strength) * seq_emb

                seq_emb_1 = self.seq_gate(state=seq_emb, input_features=refined_seq)
            elif self.use_simple_conf_gated_fusion and self.simple_conf_gate is not None:
                # Baseline 2: direct pLDDT-weighted fusion (no cross-attention)
                seq_emb_1 = self.simple_conf_gate(seq_emb, struct_map, data.plddt)
            else:
                seq_emb_1 = seq_emb
        else:
            seq_emb_1 = seq_emb

        # === 6. Final Fusion ===
        # FIX: Use struct_map (pLDDT-gated, RGAT-processed) as the structural stream,
        #      not raw struct_emb. This ensures earlier modules' outputs flow into fusion.
        # FIX: SGFN now refines struct_map (calibrated structure) using SCGC-calibrated seq.
        # FIX: When use_sgfn=False, checked_struct uses identity (struct_map) instead of
        #      duplicating raw struct_emb, eliminating the redundant-stream artifact.
        # FIX (round 6): SGFN uses neighbour-averaged pLDDT as its confidence cue so
        #      the SGFN signal is orthogonal to the per-residue pLDDT used by the gate
        #      module — keeps each module non-redundant.
        if self.use_structure:
            structural_stream = struct_map  # pLDDT-gated + RGAT-processed
        else:
            structural_stream = torch.zeros_like(seq_emb_1)

        sgfn_conf_key = neighbour_plddt if (
            self.use_structure and self.use_sgfn_conf_aware and isinstance(self.struct_checker, SCGCCrossAttention)
        ) else None

        if self.calibration_order == 'calibrate_then_fuse':
            # Standard: calibrate structure then fuse
            checked_struct = structural_stream  # default identity
            if self.use_structure and self.use_sgfn:
                if sgfn_conf_key is not None:
                    checked_struct = self.struct_checker(
                        query=seq_emb_1,
                        key_value=struct_map,
                        query_batch=batch_index,
                        key_value_batch=batch_index,
                        plddt_key=sgfn_conf_key,
                    )
                else:
                    checked_struct = self.struct_checker(
                        query=seq_emb_1,
                        key_value=struct_map,
                        query_batch=batch_index,
                        key_value_batch=batch_index,
                    )
            elif self.use_structure:
                checked_struct = structural_stream
            else:
                checked_struct = torch.zeros_like(seq_emb_1)

            combined_features = torch.cat([
                structural_stream,
                seq_emb_1,
                checked_struct,
            ], dim=-1)

        else:  # fuse_then_calibrate
            checked_struct = structural_stream
            if self.fusion_type == 'concat':
                combined_features = torch.cat([
                    structural_stream,
                    seq_emb_1,
                    structural_stream,
                ], dim=-1)
            else:
                if self.use_structure:
                    if sgfn_conf_key is not None:
                        checked_struct = self.struct_checker(
                            query=seq_emb_1,
                            key_value=struct_map,
                            query_batch=batch_index,
                            key_value_batch=batch_index,
                            plddt_key=sgfn_conf_key,
                        )
                    else:
                        checked_struct = self.struct_checker(
                            query=seq_emb_1,
                            key_value=struct_map,
                            query_batch=batch_index,
                            key_value_batch=batch_index,
                        )
                else:
                    checked_struct = torch.zeros_like(seq_emb_1)
                combined_features = torch.cat([
                    structural_stream,
                    seq_emb_1,
                    checked_struct,
                ], dim=-1)

        # === 6.5 Reliability scoring (drives Bi-Mamba residual, Round 7 redesign) ===
        # Round 7: reliability = q_prior(plddt_per_residue) * c_resid([mean_gate, cos_align, neighbour_conf])
        #   - q_prior is a learnable monotone sigmoid in pLDDT (eliminates the
        #     round-6 reliability inversion where low-pLDDT residues received
        #     higher reliability than high-pLDDT ones).
        #   - c_resid is a bounded MLP that can only modulate, never flip, the
        #     monotone prior.
        # Three orthogonal signals feeding c_resid:
        #   * mean_gate     — pLDDTGating's per-residue scalar gate
        #   * cos_align     — cosine similarity between calibrated seq and struct
        #   * neighbour_conf — locally-averaged pLDDT (graph-smoothed)
        if self.use_structure and gate_per_residue is not None and neighbour_plddt is not None:
            mean_gate = gate_per_residue.view(-1, 1)
            cos_align = F.cosine_similarity(seq_emb_1, struct_map, dim=-1).unsqueeze(-1)
            neighbour_conf = (neighbour_plddt / 100.0).clamp(0.0, 1.0).unsqueeze(-1)
            reliability_in = torch.cat([mean_gate, cos_align, neighbour_conf], dim=-1)  # [N, 3]

            plddt_norm = (data.plddt / 100.0).clamp(0.0, 1.0).unsqueeze(-1)              # [N, 1]
            q_prior = torch.sigmoid(self.reliability_q_scale * (plddt_norm - self.reliability_q_bias))  # [N, 1]
            c_resid_r = self.reliability_mlp(reliability_in)                              # [N, 1] ∈ (0, 1)
            c_resid_a = self.alpha_proj(reliability_in)                                   # [N, 1] ∈ (0, 1)

            reliability = q_prior * c_resid_r        # [N, 1] monotone in plddt
            alpha = q_prior * c_resid_a              # [N, 1] monotone in plddt
        else:
            reliability_in = None
            reliability = None
            alpha = None

        # === 7. Context Modeling (Bi-Mamba with reliability-aware residual) ===
        # Round 7 v2 fix: apply reliability damping to the structural sub-streams
        # UNCONDITIONALLY (independent of `use_bimamba`). Previously the damping
        # was tied to the mamba branch, which meant `wo_bimamba` quietly removed
        # BOTH the long-range scan AND the noise-gated residual stream — turning
        # the ablation into a smaller, less-regularised model that occasionally
        # generalised better (e.g. seed-123 outlier 0.7600). After the fix,
        # `wo_bimamba` is a clean removal of the long-range scan only; every
        # ablation receives the same reliability prior so the comparison is
        # apples-to-apples.
        if reliability is not None:
            struct_slice = combined_features[..., :self.hidden_dim] * reliability
            seq_slice = combined_features[..., self.hidden_dim:2 * self.hidden_dim]
            checked_slice = combined_features[..., 2 * self.hidden_dim:] * reliability
            damped_features = torch.cat([struct_slice, seq_slice, checked_slice], dim=-1)
        else:
            damped_features = combined_features

        if self.use_bimamba and self.mamba_layer is not None:
            mamba_out = self.mamba_layer(damped_features, batch=data.batch)
            if alpha is not None:
                fused_features = alpha * mamba_out + (1.0 - alpha) * damped_features
            else:
                fused_features = mamba_out
        else:
            fused_features = damped_features

        fused_node_embedding = self.final_projection(fused_features)

        # === 8. Quality-routed Fusion (Round 7) ===
        # Three node-level experts pooled independently; a graph-level router
        # decides how to mix them based on the graph's pLDDT statistics. This
        # gives Fusion an explicit specialization story (E_seq -> low,
        # E_mid -> mid, E_high -> high), removing the round-6 reliance on a
        # single black-box concat MLP.
        if self.use_structure:
            node_balanced = self.expert_balanced_proj(
                torch.cat([seq_emb_1, struct_map], dim=-1)
            )
        else:
            node_balanced = seq_emb_1

        z_seq = self.global_pooling(seq_emb_1, data.batch)
        z_mid = self.global_pooling(node_balanced, data.batch)
        z_high = self.global_pooling(fused_node_embedding, data.batch)

        if self.use_structure:
            # Per-graph pLDDT statistics in [0, 1] / [0, 1] / [0, 1] ranges so
            # the router input lives on a comparable scale.
            plddt_norm_pr = (data.plddt / 100.0).clamp(0.0, 1.0)
            graph_mean = scatter_mean(plddt_norm_pr, data.batch, dim=0)                       # [B]
            graph_low = scatter_mean((plddt_norm_pr < 0.7).float(), data.batch, dim=0)        # [B]
            graph_sq = scatter_mean(plddt_norm_pr ** 2, data.batch, dim=0)
            graph_var = (graph_sq - graph_mean ** 2).clamp(min=0.0)                           # [B]
            graph_q = torch.stack([graph_mean, graph_low, graph_var], dim=-1)                 # [B, 3]
            router_logits = self.quality_router(graph_q)
            w = F.softmax(router_logits, dim=-1)                                              # [B, 3]
        else:
            # Without structure all experts collapse to the seq stream.
            B = z_seq.size(0)
            w = torch.tensor([1.0, 0.0, 0.0], device=z_seq.device).expand(B, 3)
            router_logits = None

        global_embedding = (
            w[:, 0:1] * z_seq
            + w[:, 1:2] * z_mid
            + w[:, 2:3] * z_high
        )
        activity_pred = self.activity_predictor(global_embedding)

        # FIX v2 (Codex Round 5): unify Stage1 alignment anchor across
        # ablations to keep the contrastive objective fair. The anchor is
        # always the structural stream that feeds the final fusion:
        #   * with SGFN  -> checked_struct (cross-modal refined)
        #   * without SGFN -> structural_stream (identity passthrough)
        # This way every ablation aligns against its own "final structure
        # representation"; sgfn_concat is no longer penalized by a missing
        # second anchor.
        if self.use_structure:
            struct_anchor = checked_struct
        else:
            struct_anchor = structural_stream  # zeros when wo_structure
        return {
            'activity_pred': activity_pred,
            'seq_global': self.global_pooling(seq_emb_1, data.batch),
            'struct_global': self.global_pooling(struct_anchor, data.batch),
            # raw RGVP+pos-enc retained for P0-1 perturbation diagnostics
            'struct_raw_global': self.global_pooling(struct_emb, data.batch),
            'combined_global': global_embedding,
            'fused_node_features': fused_node_embedding,
            # Per-residue scalar gate + pLDDT for diagnostics + monotonicity loss.
            'gate_per_residue': gate_per_residue,
            'plddt_per_residue': data.plddt if self.use_structure else None,
            'reliability_per_residue': reliability,
            # Round 7 diagnostics
            'scgc_strength_per_residue': scgc_strength,
            'router_weights_per_graph': w if self.use_structure else None,
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