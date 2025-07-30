#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
评估指标计算工具
包含所有用于模型评估的指标计算函数
"""

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, matthews_corrcoef,
    confusion_matrix, classification_report
)
from typing import Dict, Any, Tuple, Union, Optional
import logging

logger = logging.getLogger(__name__)


def calculate_metrics(y_true: Union[np.ndarray, torch.Tensor], 
                     y_pred: Union[np.ndarray, torch.Tensor],
                     y_scores: Optional[Union[np.ndarray, torch.Tensor]] = None,
                     threshold: float = 0.550) -> Dict[str, float]:
    """
    计算分类任务的所有评估指标
    
    Args:
        y_true: 真实标签
        y_pred: 预测标签 (如果y_scores为None) 或预测概率 (如果y_scores为None且y_pred为概率)
        y_scores: 预测概率分数 (可选)
        threshold: 二分类阈值
        
    Returns:
        包含所有指标的字典
    """
    # 转换为numpy数组
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    if y_scores is not None and isinstance(y_scores, torch.Tensor):
        y_scores = y_scores.detach().cpu().numpy()
    
    # 确保数据类型正确
    y_true = y_true.astype(int)
    
    # 如果没有提供y_scores，但y_pred看起来像概率，则使用y_pred作为概率
    if y_scores is None:
        if np.issubdtype(y_pred.dtype, np.floating) and np.all((y_pred >= 0) & (y_pred <= 1)):
            y_scores = y_pred
            y_pred = (y_pred > threshold).astype(int)
        else:
            y_pred = y_pred.astype(int)
    else:
        # 使用阈值将概率转换为预测标签
        y_pred = (y_scores > threshold).astype(int)
    
    try:
        # 检查标签的唯一值数量
        unique_labels = np.unique(y_true)
        n_classes = len(unique_labels)
        
        # 基本分类指标
        accuracy = accuracy_score(y_true, y_pred)
        
        # 根据类别数量选择合适的average参数
        if n_classes == 2:
            # 二分类任务
            precision = precision_score(y_true, y_pred, average='binary', zero_division=0)
            recall = recall_score(y_true, y_pred, average='binary', zero_division=0)
            f1 = f1_score(y_true, y_pred, average='binary', zero_division=0)
        else:
            # 多分类任务
            precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
            recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
            f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        
        mcc = matthews_corrcoef(y_true, y_pred)
        
        # 混淆矩阵相关指标
        if n_classes == 2:
            # 二分类混淆矩阵
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # 等同于recall
        else:
            # 多分类情况，设置默认值
            tp = tn = fp = fn = 0
            specificity = sensitivity = 0.0
        
        metrics = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'sensitivity': float(sensitivity),
            'specificity': float(specificity),
            'f1': float(f1),
            'mcc': float(mcc),
            'tp': int(tp),
            'tn': int(tn),
            'fp': int(fp),
            'fn': int(fn)
        }
        
        # 概率相关指标（需要概率分数）
        if y_scores is not None:
            try:
                auc_roc = roc_auc_score(y_true, y_scores)
                auc_pr = average_precision_score(y_true, y_scores)
                metrics.update({
                    'auc': float(auc_roc),
                    'auc_roc': float(auc_roc),
                    'aupr': float(auc_pr),
                    'auc_pr': float(auc_pr)
                })
            except ValueError as e:
                logger.warning(f"无法计算AUC指标: {e}")
                metrics.update({
                    'auc': 0.0,
                    'auc_roc': 0.0,
                    'aupr': 0.0,
                    'auc_pr': 0.0
                })
        else:
            metrics.update({
                'auc': 0.0,
                'auc_roc': 0.0,
                'aupr': 0.0,
                'auc_pr': 0.0
            })
            
        return metrics
        
    except Exception as e:
        logger.error(f"计算指标时出错: {e}")
        # 返回默认指标
        return {
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'sensitivity': 0.0,
            'specificity': 0.0,
            'f1': 0.0,
            'mcc': 0.0,
            'auc': 0.0,
            'auc_roc': 0.0,
            'aupr': 0.0,
            'auc_pr': 0.0,
            'tp': 0,
            'tn': 0,
            'fp': 0,
            'fn': 0
        }


def calculate_loss_metrics(y_true: Union[np.ndarray, torch.Tensor],
                          y_pred: Union[np.ndarray, torch.Tensor]) -> Dict[str, float]:
    """
    计算损失相关指标
    
    Args:
        y_true: 真实值
        y_pred: 预测值
        
    Returns:
        损失指标字典
    """
    # 转换为numpy数组
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    
    try:
        # 均方误差
        mse = np.mean((y_true - y_pred) ** 2)
        # 平均绝对误差
        mae = np.mean(np.abs(y_true - y_pred))
        # 均方根误差
        rmse = np.sqrt(mse)
        
        return {
            'mse': float(mse),
            'mae': float(mae),
            'rmse': float(rmse)
        }
    except Exception as e:
        logger.error(f"计算损失指标时出错: {e}")
        return {
            'mse': 0.0,
            'mae': 0.0,
            'rmse': 0.0
        }


def print_metrics_summary(metrics: Dict[str, float], 
                         phase: str = "Test",
                         logger_obj: Optional[logging.Logger] = None) -> None:
    """
    打印指标摘要
    
    Args:
        metrics: 指标字典
        phase: 阶段名称 (Train/Val/Test)
        logger_obj: 日志对象
    """
    if logger_obj is None:
        logger_obj = logger
    
    logger_obj.info(f"\n=== {phase} 指标摘要 ===")
    logger_obj.info(f"准确率 (Accuracy): {metrics.get('accuracy', 0):.4f}")
    logger_obj.info(f"精确率 (Precision): {metrics.get('precision', 0):.4f}")
    logger_obj.info(f"召回率 (Recall): {metrics.get('recall', 0):.4f}")
    logger_obj.info(f"F1分数: {metrics.get('f1', 0):.4f}")
    logger_obj.info(f"MCC: {metrics.get('mcc', 0):.4f}")
    logger_obj.info(f"特异性 (Specificity): {metrics.get('specificity', 0):.4f}")
    
    if metrics.get('auc', 0) > 0:
        logger_obj.info(f"AUC-ROC: {metrics.get('auc', 0):.4f}")
        logger_obj.info(f"AUC-PR: {metrics.get('aupr', 0):.4f}")
    
    if any(key in metrics for key in ['tp', 'tn', 'fp', 'fn']):
        logger_obj.info(f"混淆矩阵: TP={metrics.get('tp', 0)}, TN={metrics.get('tn', 0)}, "
                       f"FP={metrics.get('fp', 0)}, FN={metrics.get('fn', 0)}")


def get_best_threshold(y_true: Union[np.ndarray, torch.Tensor],
                      y_scores: Union[np.ndarray, torch.Tensor],
                      metric: str = 'f1') -> Tuple[float, float]:
    """
    寻找最佳阈值
    
    Args:
        y_true: 真实标签
        y_scores: 预测概率
        metric: 优化的指标 ('f1', 'mcc', 'precision', 'recall')
        
    Returns:
        (最佳阈值, 最佳指标值)
    """
    # 转换为numpy数组
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_scores, torch.Tensor):
        y_scores = y_scores.detach().cpu().numpy()
    
    thresholds = np.linspace(0.01, 0.99, 99)
    best_threshold = 0.550  # 默认使用优化后的阈值
    best_score = 0.0
    
    for threshold in thresholds:
        y_pred = (y_scores > threshold).astype(int)
        
        try:
            if metric == 'f1':
                score = f1_score(y_true, y_pred, zero_division=0)
            elif metric == 'mcc':
                score = matthews_corrcoef(y_true, y_pred)
            elif metric == 'precision':
                score = precision_score(y_true, y_pred, zero_division=0)
            elif metric == 'recall':
                score = recall_score(y_true, y_pred, zero_division=0)
            else:
                raise ValueError(f"不支持的指标: {metric}")
                
            if score > best_score:
                best_score = score
                best_threshold = threshold
                
        except Exception:
            continue
    
    return best_threshold, best_score


def format_metrics_for_logging(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """
    格式化指标用于日志记录
    
    Args:
        metrics: 原始指标字典
        
    Returns:
        格式化后的指标字典
    """
    formatted = {}
    
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            if isinstance(value, float):
                formatted[key] = round(value, 4)
            else:
                formatted[key] = value
        else:
            formatted[key] = value
    
    return formatted
