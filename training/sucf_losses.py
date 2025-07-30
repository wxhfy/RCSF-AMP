"""
SGG-Net专用损失函数
支持两阶段训练的动态损失权重和多种对比损失
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, List
import logging

logger = logging.getLogger(__name__)


class AlignmentContrastiveLoss(nn.Module):
    """
    模态对齐对比损失
    用于第一阶段的序列-结构特征对齐
    """
    def __init__(self, temperature=0.1, reduction='mean'):
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction
    
    def forward(self, seq_features, struct_features):
        """
        计算模态对齐对比损失
        
        Args:
            seq_features: [B, dim] 序列全局特征
            struct_features: [B, dim] 结构全局特征
            
        Returns:
            loss: 对比损失值
        """
        # 标准化特征
        seq_features = F.normalize(seq_features, p=2, dim=1)
        struct_features = F.normalize(struct_features, p=2, dim=1)
        
        # 计算相似度矩阵
        similarity = torch.matmul(seq_features, struct_features.T) / self.temperature
        
        batch_size = seq_features.size(0)
        labels = torch.arange(batch_size, device=seq_features.device)
        
        # 计算交叉熵损失 (双向)
        loss_seq_to_struct = F.cross_entropy(similarity, labels)
        loss_struct_to_seq = F.cross_entropy(similarity.T, labels)
        
        loss = (loss_seq_to_struct + loss_struct_to_seq) / 2
        
        return loss


class SupervisedContrastiveLoss(nn.Module):
    """
    有监督对比学习损失
    用于第二阶段的标签感知特征学习
    """
    def __init__(self, temperature=0.07, reduction='mean'):
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction
    
    def forward(self, features, labels):
        """
        计算有监督对比损失
        
        Args:
            features: [B, dim] 特征向量
            labels: [B] 标签或DataBatch对象
            
        Returns:
            loss: 有监督对比损失
        """
        device = features.device
        batch_size = features.size(0)
        
        # 标准化特征
        features = F.normalize(features, p=2, dim=1)
        
        # 计算相似度矩阵
        similarity_matrix = torch.matmul(features, features.T)
        
        # 安全提取标签tensor
        if hasattr(labels, 'y') and labels.y is not None:
            labels_tensor = labels.y
        elif torch.is_tensor(labels):
            labels_tensor = labels
        else:
            # 如果labels是其他类型，尝试转换
            labels_tensor = torch.tensor(labels, device=device)
        
        # 创建标签掩码
        labels_tensor = labels_tensor.contiguous().view(-1, 1)
        mask = torch.eq(labels_tensor, labels_tensor.T).float().to(device)
        
        # 移除自身对角线
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask
        
        # 计算对比损失
        exp_logits = torch.exp(similarity_matrix / self.temperature)
        exp_logits = exp_logits * logits_mask
        
        log_prob = similarity_matrix / self.temperature - torch.log(exp_logits.sum(1, keepdim=True))
        
        # 计算正样本对的平均log概率
        mask_sum = mask.sum(1)
        valid_samples = mask_sum > 0
        
        if valid_samples.sum() == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        mean_log_prob_pos = torch.zeros_like(mask_sum)
        mean_log_prob_pos[valid_samples] = (mask * log_prob).sum(1)[valid_samples] / mask_sum[valid_samples]
        
        loss = -mean_log_prob_pos[valid_samples]
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class SUCFTotalLoss(nn.Module):
    """
    SGG-Net总损失函数
    支持两阶段训练的动态损失权重
    """
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        loss_config = config.get('training', {}).get('loss_config', {})
        
        # 初始化各种损失函数
        self.activity_loss = nn.BCEWithLogitsLoss()
        
        self.alignment_contrastive_loss = AlignmentContrastiveLoss(
            temperature=loss_config.get('alignment_contrastive_temperature', 0.1)
        )
        
        self.supervised_contrastive_loss = SupervisedContrastiveLoss(
            temperature=loss_config.get('supervised_contrastive_temperature', 0.07)
        )
        
        # 门控正则化配置
        gating_config = loss_config.get('gating_regularization', {})
        self.gating_reg_enabled = gating_config.get('enable', True)
        self.gating_reg_alpha = gating_config.get('alpha', 0.05)
        self.gating_reg_target_entropy = gating_config.get('target_entropy', 0.5)
        
        logger.info("SGG-Net总损失函数初始化完成")
    
    def compute_gating_regularization(self, model_output):
        """
        计算门控正则化损失
        鼓励注意力权重的熵接近目标值
        """
        if not self.gating_reg_enabled:
            return torch.tensor(0.0, device=next(iter(model_output.values())).device)
        
        reg_loss = torch.tensor(0.0, device=next(iter(model_output.values())).device)
        
        # 如果模型输出中包含注意力权重，计算熵正则化
        for key in ['structure_map', 'refined_seq_features']:
            if key in model_output:
                attention_weights = model_output[key]
                # 计算softmax归一化的注意力权重
                attention_probs = F.softmax(attention_weights, dim=-1)
                # 计算熵
                entropy = -(attention_probs * torch.log(attention_probs + 1e-8)).sum(dim=-1).mean()
                # L2损失，鼓励熵接近目标值
                entropy_loss = (entropy - self.gating_reg_target_entropy) ** 2
                reg_loss = reg_loss + entropy_loss
        
        return self.gating_reg_alpha * reg_loss
    
    def forward(self, model_output, targets, stage_info):
        """
        计算总损失
        
        Args:
            model_output: 模型输出字典
            targets: 目标标签
            stage_info: 阶段信息，包含active_losses和loss_weights
            
        Returns:
            total_loss: 总损失
            loss_dict: 各项损失的详细信息
        """
        # 安全获取device - 从targets的某个tensor属性获取
        if hasattr(targets, 'y') and targets.y is not None:
            device = targets.y.device
        elif hasattr(targets, 'x') and targets.x is not None:
            device = targets.x.device
        else:
            # 从model_output中获取device
            if 'activity_pred' in model_output:
                device = model_output['activity_pred'].device
            else:
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 获取阶段配置
        active_losses = stage_info.get('active_losses', ['activity'])
        loss_weights = stage_info.get('loss_weights', {'activity': 1.0})
        
        loss_dict = {}
        total_loss = torch.tensor(0.0, device=device)
        
        # 1. 活性预测损失
        if 'activity' in active_losses:
            activity_pred = model_output['activity_pred'].squeeze()
            # 安全获取标签并转换为float
            if hasattr(targets, 'y') and targets.y is not None:
                target_labels = targets.y.float()
            else:
                # 如果没有y属性，尝试直接使用targets
                target_labels = targets.float() if torch.is_tensor(targets) else targets
            
            activity_loss = self.activity_loss(activity_pred, target_labels)
            loss_dict['activity'] = activity_loss.item()
            
            weight = loss_weights.get('activity', 0.0)
            total_loss = total_loss + weight * activity_loss
        
        # 2. 模态对齐对比损失
        if 'alignment_contrastive' in active_losses:
            seq_global = model_output.get('seq_global')
            struct_global = model_output.get('struct_global')
            
            if seq_global is not None and struct_global is not None:
                alignment_loss = self.alignment_contrastive_loss(seq_global, struct_global)
                loss_dict['alignment_contrastive'] = alignment_loss.item()
                
                weight = loss_weights.get('alignment_contrastive', 0.0)
                total_loss = total_loss + weight * alignment_loss
            else:
                logger.warning("模态对齐损失需要seq_global和struct_global特征")
        
        # 3. 有监督对比损失
        if 'supervised_contrastive' in active_losses:
            combined_global = model_output.get('combined_global')
            
            if combined_global is not None:
                sup_contrastive_loss = self.supervised_contrastive_loss(combined_global, targets)
                loss_dict['supervised_contrastive'] = sup_contrastive_loss.item()
                
                weight = loss_weights.get('supervised_contrastive', 0.0)
                total_loss = total_loss + weight * sup_contrastive_loss
            else:
                logger.warning("有监督对比损失需要combined_global特征")
        
        # 4. 门控正则化损失
        gating_reg_loss = self.compute_gating_regularization(model_output)
        loss_dict['gating_regularization'] = gating_reg_loss.item()
        total_loss = total_loss + gating_reg_loss
        
        # 记录总损失
        loss_dict['total'] = total_loss.item()
        loss_dict['total_loss'] = total_loss  # 添加total_loss键用于训练
        
        return loss_dict


def create_sucf_loss_function(config):
    """
    工厂函数：创建SGG-Net损失函数
    
    Args:
        config: 训练配置字典
        
    Returns:
        loss_fn: SUCF损失函数实例
    """
    loss_fn = SUCFTotalLoss(config)

    logger.info("SUCF损失函数创建完成")
    logger.info(f"  模态对齐温度: {loss_fn.alignment_contrastive_loss.temperature}")
    logger.info(f"  监督对比温度: {loss_fn.supervised_contrastive_loss.temperature}")
    logger.info(f"  门控正则化: {loss_fn.gating_reg_enabled}")
    
    return loss_fn


if __name__ == "__main__":
    # 测试代码
    print("Testing SGG-Net loss functions...")
    
    # 测试模态对齐损失
    alignment_loss = AlignmentContrastiveLoss(temperature=0.1)
    seq_feat = torch.randn(4, 128)
    struct_feat = torch.randn(4, 128)
    loss1 = alignment_loss(seq_feat, struct_feat)
    print(f"Alignment loss: {loss1.item():.4f}")
    
    # 测试有监督对比损失
    sup_loss = SupervisedContrastiveLoss(temperature=0.07)
    features = torch.randn(4, 128)
    labels = torch.tensor([0, 1, 0, 1])
    loss2 = sup_loss(features, labels)
    print(f"Supervised contrastive loss: {loss2.item():.4f}")
    
    print("All loss functions tested successfully!")
