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


class SUCFTotalLoss(nn.Module):
    """
    Total loss function for the SUCF model, supporting dynamic weighting for two-stage training.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        loss_config = config.get('training', {}).get('loss_config', {})
        
        # Initialize individual loss functions
        self.activity_loss = nn.BCEWithLogitsLoss()
        self.alignment_contrastive_loss = AlignmentContrastiveLoss(
            temperature=loss_config.get('alignment_contrastive_temperature', 0.1)
        )
        self.supervised_contrastive_loss = SupervisedContrastiveLoss(
            temperature=loss_config.get('supervised_contrastive_temperature', 0.07)
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
            
            activity_loss_val = self.activity_loss(activity_pred, target_labels)
            loss_dict['activity'] = activity_loss_val.item()
            total_loss += loss_weights.get('activity', 0.0) * activity_loss_val
        
        # 2. Modal Alignment Contrastive Loss
        if 'alignment_contrastive' in active_losses:
            seq_global = model_output.get('seq_global')
            struct_global = model_output.get('struct_global')
            
            if seq_global is not None and struct_global is not None:
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