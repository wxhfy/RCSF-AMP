"""
Loss functions for the SUCF model.
Supports dynamic loss weighting for two-stage training and various contrastive losses.
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict
import logging

logger = logging.getLogger(__name__)


class AlignmentContrastiveLoss(nn.Module):
    """
    Modal Alignment Contrastive Loss (InfoNCE style).
    Used for aligning sequence and structure features in the first training stage.
    """
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, seq_features, struct_features):
        """
        Calculates the modal alignment contrastive loss.
        
        Args:
            seq_features (torch.Tensor): Global sequence features of shape [B, D].
            struct_features (torch.Tensor): Global structure features of shape [B, D].
            
        Returns:
            torch.Tensor: The contrastive loss value.
        """
        # Normalize features to lie on the unit hypersphere
        seq_features = F.normalize(seq_features, p=2, dim=1)
        struct_features = F.normalize(struct_features, p=2, dim=1)
        
        # Calculate cosine similarity matrix
        similarity = torch.matmul(seq_features, struct_features.T) / self.temperature
        
        batch_size = seq_features.size(0)
        labels = torch.arange(batch_size, device=seq_features.device)
        
        # Calculate cross-entropy loss in both directions (sequence-to-structure and vice-versa)
        loss_seq_to_struct = F.cross_entropy(similarity, labels)
        loss_struct_to_seq = F.cross_entropy(similarity.T, labels)
        
        loss = (loss_seq_to_struct + loss_struct_to_seq) / 2
        return loss


class SupervisedContrastiveLoss(nn.Module):
    """
    Supervised Contrastive Learning Loss.
    Used for label-aware feature learning in the second training stage.
    """
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, features, labels):
        """
        Calculates the supervised contrastive loss.
        
        Args:
            features (torch.Tensor): Feature vectors of shape [B, D].
            labels (torch.Tensor or DataBatch): Labels for the features.
            
        Returns:
            torch.Tensor: The supervised contrastive loss value.
        """
        device = features.device
        batch_size = features.size(0)
        
        features = F.normalize(features, p=2, dim=1)
        similarity_matrix = torch.matmul(features, features.T)
        
        # Extract label tensor safely
        labels_tensor = labels.y if hasattr(labels, 'y') and labels.y is not None else labels
        
        # Create a mask for positive pairs (samples with the same label)
        labels_tensor = labels_tensor.contiguous().view(-1, 1)
        mask = torch.eq(labels_tensor, labels_tensor.T).float().to(device)
        
        # Mask out self-comparisons (diagonal elements)
        logits_mask = ~torch.eye(batch_size, dtype=torch.bool, device=device)
        mask = mask * logits_mask
        
        # Compute the contrastive loss
        exp_logits = torch.exp(similarity_matrix / self.temperature) * logits_mask
        log_prob = (similarity_matrix / self.temperature) - torch.log(exp_logits.sum(1, keepdim=True))
        
        # Compute the mean log-probability for positive pairs
        mask_sum = mask.sum(1)
        valid_samples_mask = mask_sum > 0
        if not valid_samples_mask.any():
            return torch.tensor(0.0, device=device, requires_grad=True)
            
        mean_log_prob_pos = (mask * log_prob).sum(1)[valid_samples_mask] / mask_sum[valid_samples_mask]
        
        loss = -mean_log_prob_pos.mean()
        return loss


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance and hard example mining.
    FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
    """
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        probs = torch.sigmoid(inputs)
        pt = torch.where(targets == 1, probs, 1 - probs)
        focal_weight = (1 - pt) ** self.gamma
        alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        loss = alpha_t * focal_weight * bce_loss
        return loss.mean()


class GateMonotonicityLoss(nn.Module):
    """Encourages the pLDDT gate to be monotonically non-decreasing across pLDDT bins.

    The gate is expected to be near 0 for very-low pLDDT (use sequence) and
    near 1 for high pLDDT (use structure). A simple ReLU-margin loss penalises
    cases where the bin-wise mean gate fails to increase by at least `margin`
    between consecutive bins. This is a much safer regulariser than maximising
    raw variance — it directly aligns the gate with the desired pLDDT-conditioned
    behaviour and is robust to per-channel modulation.
    """

    DEFAULT_BINS = ((0.0, 50.0), (50.0, 70.0), (70.0, 90.0), (90.0, 101.0))

    def __init__(self, margin: float = 0.05, bins=None):
        super().__init__()
        self.margin = float(margin)
        self.bins = list(bins) if bins is not None else list(self.DEFAULT_BINS)

    def forward(self, gate_per_residue, plddt):
        """
        Args:
            gate_per_residue (torch.Tensor): Scalar gate per residue [N, 1] or [N].
            plddt (torch.Tensor): Raw pLDDT per residue [N] or [N, 1].

        Returns:
            torch.Tensor: Scalar monotonicity loss.
        """
        if gate_per_residue is None or plddt is None:
            return torch.tensor(0.0, device=plddt.device if plddt is not None else 'cpu')

        gate_flat = gate_per_residue.view(-1)
        plddt_flat = plddt.view(-1)

        bin_means = []
        for lo, hi in self.bins:
            mask = (plddt_flat >= lo) & (plddt_flat < hi)
            if mask.any():
                bin_means.append(gate_flat[mask].mean())

        if len(bin_means) < 2:
            return torch.tensor(0.0, device=gate_flat.device)

        means = torch.stack(bin_means)
        diffs = means[1:] - means[:-1]
        # Penalise any pair that does not rise by at least `margin`.
        return F.relu(self.margin - diffs).mean()


class ReliabilityRegularizationLoss(nn.Module):
    """Encourages c_resid (MLP modulation) to stay close to 1.0 so the reliability
    score closely follows the monotone q_prior. This prevents the MLP from learning
    spurious patterns that break pLDDT monotonicity and introduce training variance.
    """

    def __init__(self, margin: float = 0.3):
        super().__init__()
        self.margin = float(margin)

    def forward(self, c_resid_r, c_resid_a):
        if c_resid_r is None or c_resid_a is None:
            return torch.tensor(0.0)
        # Penalise deviation from 1.0, but allow up to `margin` free.
        loss_r = F.relu((1.0 - c_resid_r).abs() - self.margin).mean()
        loss_a = F.relu((1.0 - c_resid_a).abs() - self.margin).mean()
        return loss_r + loss_a


class SUCFTotalLoss(nn.Module):
    """
    Total loss function for the SUCF model, supporting dynamic weighting for two-stage training.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        loss_config = config.get('training', {}).get('loss_config', {})

        # Initialize individual loss functions
        self.use_focal = loss_config.get('use_focal_loss', False)
        self.use_class_weight = loss_config.get('use_class_weight', False)
        self.pos_weight = loss_config.get('pos_weight', 1.0)
        self.label_smoothing = loss_config.get('label_smoothing', 0.0)

        if self.use_focal:
            self.activity_loss = FocalLoss(
                alpha=loss_config.get('focal_alpha', 0.25),
                gamma=loss_config.get('focal_gamma', 2.0)
            )
            logger.info(f"Using Focal Loss (alpha={loss_config.get('focal_alpha', 0.25)}, gamma={loss_config.get('focal_gamma', 2.0)})")
        elif self.use_class_weight:
            self.activity_loss = None  # Will handle in forward with proper device
            logger.info(f"Using Weighted BCE (pos_weight={self.pos_weight})")
        else:
            self.activity_loss = nn.BCEWithLogitsLoss()
        self.alignment_contrastive_loss = AlignmentContrastiveLoss(
            temperature=loss_config.get('alignment_contrastive_temperature', 0.1)
        )
        self.supervised_contrastive_loss = SupervisedContrastiveLoss(
            temperature=loss_config.get('supervised_contrastive_temperature', 0.07)
        )
        self.gate_monotonicity_loss = GateMonotonicityLoss(
            margin=loss_config.get('gate_monotonicity_margin', 0.05)
        )
        self.reliability_reg_loss = ReliabilityRegularizationLoss(
            margin=loss_config.get('reliability_reg_margin', 0.3)
        )
        logger.info("SUCF total loss function initialized.")
    
    def forward(self, model_output: Dict, targets, stage_info: Dict) -> Dict:
        """
        Calculates the total loss based on the current training stage.
        
        Args:
            model_output: The dictionary output from the SUCF model.
            targets: The ground truth labels or DataBatch object.
            stage_info: A dictionary specifying active losses and their weights.
            
        Returns:
            A dictionary containing the total loss and a breakdown of individual losses.
        """
        device = model_output['activity_pred'].device
        
        active_losses = stage_info.get('active_losses', ['activity'])
        loss_weights = stage_info.get('loss_weights', {'activity': 1.0})
        
        loss_dict = {}
        total_loss = torch.tensor(0.0, device=device)
        
        # 1. Activity Prediction Loss (BCE)
        if 'activity' in active_losses:
            activity_pred = model_output['activity_pred'].squeeze()
            target_labels = targets.y.float()

            # Apply label smoothing if configured
            if self.label_smoothing > 0:
                target_labels = target_labels * (1 - self.label_smoothing) + 0.5 * self.label_smoothing

            if self.use_class_weight:
                # Weighted BCE with proper device placement
                weight = torch.tensor([self.pos_weight], device=device)
                activity_loss_val = F.binary_cross_entropy_with_logits(
                    activity_pred, target_labels, pos_weight=weight
                )
            else:
                activity_loss_val = self.activity_loss(activity_pred, target_labels)
            loss_dict['activity'] = activity_loss_val.item()
            total_loss += loss_weights.get('activity', 0.0) * activity_loss_val
        
        # 2. Modal Alignment Contrastive Loss
        if 'alignment_contrastive' in active_losses:
            seq_global = model_output.get('seq_global')
            struct_global = model_output.get('struct_global')

            if seq_global is not None and struct_global is not None:
                # Unified single-anchor InfoNCE: aligns calibrated sequence to
                # the structural stream that actually feeds final fusion.  Same
                # objective shape across all ablations -> fair comparison
                # (Codex Round 5 fix).
                alignment_loss_val = self.alignment_contrastive_loss(seq_global, struct_global)
                loss_dict['alignment_contrastive'] = alignment_loss_val.item()
                total_loss += loss_weights.get('alignment_contrastive', 0.0) * alignment_loss_val
            else:
                logger.warning("Alignment loss requires 'seq_global' and 'struct_global' features.")
        
        # 3. Supervised Contrastive Loss
        if 'supervised_contrastive' in active_losses:
            combined_global = model_output.get('combined_global')
            if combined_global is not None:
                sup_contrastive_loss_val = self.supervised_contrastive_loss(combined_global, targets)
                loss_dict['supervised_contrastive'] = sup_contrastive_loss_val.item()
                total_loss += loss_weights.get('supervised_contrastive', 0.0) * sup_contrastive_loss_val
            else:
                logger.warning("Supervised contrastive loss requires 'combined_global' features.")

        # 4. Gate Monotonicity Regulariser
        if 'gate_monotonicity' in active_losses:
            gate_per_residue = model_output.get('gate_per_residue')
            plddt_per_residue = model_output.get('plddt_per_residue')
            if gate_per_residue is not None and plddt_per_residue is not None:
                mono_loss_val = self.gate_monotonicity_loss(gate_per_residue, plddt_per_residue)
                loss_dict['gate_monotonicity'] = mono_loss_val.item() if mono_loss_val.requires_grad or mono_loss_val.numel() > 0 else float(mono_loss_val)
                total_loss += loss_weights.get('gate_monotonicity', 0.0) * mono_loss_val

        # 5. Reliability Regularisation (Round 9): keep c_resid MLPs close to identity.
        if 'reliability_reg' in active_losses:
            c_resid_r = model_output.get('c_resid_r')
            c_resid_a = model_output.get('c_resid_a')
            if c_resid_r is not None and c_resid_a is not None:
                reg_loss_val = self.reliability_reg_loss(c_resid_r, c_resid_a)
                loss_dict['reliability_reg'] = (reg_loss_val.item() if hasattr(reg_loss_val, 'item')
                                                else float(reg_loss_val))
                total_loss += loss_weights.get('reliability_reg', 0.0) * reg_loss_val

        loss_dict['total_loss'] = total_loss
        return loss_dict


def create_sucf_loss_function(config):
    """
    Factory function to create the SUCF loss function module.
    
    Args:
        config: The training configuration dictionary.
        
    Returns:
        An instance of the SUCFTotalLoss function.
    """
    loss_fn = SUCFTotalLoss(config)
    logger.info("SUCF loss function created.")
    logger.info(f"  Alignment Contrastive Temperature: {loss_fn.alignment_contrastive_loss.temperature}")
    logger.info(f"  Supervised Contrastive Temperature: {loss_fn.supervised_contrastive_loss.temperature}")
    return loss_fn