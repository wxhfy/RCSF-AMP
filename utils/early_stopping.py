#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
早停工具类
用于监控训练过程中的指标变化，避免过拟合
"""

import numpy as np
import logging
from typing import Optional, Union

logger = logging.getLogger(__name__)


class EarlyStopping:
    """
    早停类，用于监控验证指标并决定是否停止训练
    """
    
    def __init__(self, 
                 patience: int = 10,
                 min_delta: float = 0.0,
                 mode: str = 'min',
                 restore_best_weights: bool = False,  # 默认改为False，避免DDP中的权重不一致
                 verbose: bool = True):
        """
        初始化早停对象
        
        Args:
            patience: 没有改善的epoch数量，超过此数量则停止训练
            min_delta: 被认为是改善的最小变化量
            mode: 'min' 表示指标越小越好，'max' 表示指标越大越好
            restore_best_weights: 是否在停止时恢复最佳权重（DDP训练时建议设为False）
            verbose: 是否打印详细信息
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        self.verbose = verbose
        
        self.wait = 0
        self.stopped_epoch = 0
        self.best_score = None
        self.early_stop = False
        self.best_weights = None
        
        if mode == 'min':
            self.monitor_op = np.less
            self.min_delta *= -1
        elif mode == 'max':
            self.monitor_op = np.greater
            self.min_delta *= 1
        else:
            raise ValueError(f"模式必须是 'min' 或 'max'，得到: {mode}")
    
    def __call__(self, 
                 current_score: float,
                 model: Optional[object] = None,
                 epoch: Optional[int] = None) -> bool:
        """
        检查是否应该早停
        
        Args:
            current_score: 当前epoch的监控指标值
            model: 模型对象（用于保存最佳权重）
            epoch: 当前epoch数
            
        Returns:
            是否应该停止训练
        """
        if self.best_score is None:
            self.best_score = current_score
            if model is not None and self.restore_best_weights:
                self.best_weights = self._get_model_weights(model)
        
        elif self.monitor_op(current_score - self.min_delta, self.best_score):
            self.best_score = current_score
            self.wait = 0
            if model is not None and self.restore_best_weights:
                self.best_weights = self._get_model_weights(model)
            
            if self.verbose:
                logger.info(f"Epoch {epoch}: 指标改善到 {current_score:.4f}")
        
        else:
            self.wait += 1
            if self.verbose and self.wait > 0:
                logger.info(f"Epoch {epoch}: 指标未改善 (当前: {current_score:.4f}, "
                          f"最佳: {self.best_score:.4f}), 等待 {self.wait}/{self.patience}")
            
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.early_stop = True
                
                if self.verbose:
                    logger.info(f"早停触发！在epoch {epoch}停止训练")
                    logger.info(f"最佳指标: {self.best_score:.4f}")
                
                # 恢复最佳权重
                if model is not None and self.restore_best_weights and self.best_weights is not None:
                    self._set_model_weights(model, self.best_weights)
                    if self.verbose:
                        logger.info("已恢复最佳模型权重")
        
        return self.early_stop
    
    def _get_model_weights(self, model):
        """获取模型权重的深拷贝"""
        try:
            import copy
            return copy.deepcopy(model.state_dict())
        except Exception as e:
            logger.warning(f"无法获取模型权重: {e}")
            return None
    
    def _set_model_weights(self, model, weights):
        """设置模型权重"""
        try:
            model.load_state_dict(weights)
        except Exception as e:
            logger.warning(f"无法设置模型权重: {e}")
    
    def reset(self):
        """重置早停状态"""
        self.wait = 0
        self.stopped_epoch = 0
        self.best_score = None
        self.early_stop = False
        self.best_weights = None
        
        if self.verbose:
            logger.info("早停状态已重置")
    
    def get_best_score(self) -> Optional[float]:
        """获取最佳分数"""
        return self.best_score
    
    def should_stop(self) -> bool:
        """检查是否应该停止"""
        return self.early_stop


class MetricTracker:
    """
    指标跟踪器，用于跟踪和记录训练过程中的指标变化
    """
    
    def __init__(self, metrics_to_track: list = None):
        """
        初始化指标跟踪器
        
        Args:
            metrics_to_track: 要跟踪的指标名称列表
        """
        self.metrics_to_track = metrics_to_track or ['loss', 'accuracy', 'f1', 'mcc']
        self.history = {metric: [] for metric in self.metrics_to_track}
        self.best_values = {}
        self.best_epochs = {}
    
    def update(self, metrics: dict, epoch: int):
        """
        更新指标历史
        
        Args:
            metrics: 当前epoch的指标字典
            epoch: 当前epoch数
        """
        for metric_name in self.metrics_to_track:
            if metric_name in metrics:
                value = metrics[metric_name]
                self.history[metric_name].append(value)
                
                # 更新最佳值
                if metric_name not in self.best_values:
                    self.best_values[metric_name] = value
                    self.best_epochs[metric_name] = epoch
                else:
                    # 根据指标类型判断是否更新最佳值
                    if self._is_better(metric_name, value, self.best_values[metric_name]):
                        self.best_values[metric_name] = value
                        self.best_epochs[metric_name] = epoch
    
    def _is_better(self, metric_name: str, current_value: float, best_value: float) -> bool:
        """
        判断当前值是否比最佳值更好
        
        Args:
            metric_name: 指标名称
            current_value: 当前值
            best_value: 最佳值
            
        Returns:
            是否更好
        """
        # 损失类指标：越小越好
        loss_metrics = ['loss', 'val_loss', 'train_loss', 'mse', 'mae', 'rmse']
        
        if any(loss_word in metric_name.lower() for loss_word in loss_metrics):
            return current_value < best_value
        else:
            # 其他指标（准确率、F1等）：越大越好
            return current_value > best_value
    
    def get_best(self, metric_name: str) -> tuple:
        """
        获取指定指标的最佳值和对应的epoch
        
        Args:
            metric_name: 指标名称
            
        Returns:
            (最佳值, 最佳epoch)
        """
        if metric_name in self.best_values:
            return self.best_values[metric_name], self.best_epochs[metric_name]
        else:
            return None, None
    
    def get_history(self, metric_name: str) -> list:
        """
        获取指定指标的历史记录
        
        Args:
            metric_name: 指标名称
            
        Returns:
            指标历史列表
        """
        return self.history.get(metric_name, [])
    
    def get_latest(self, metric_name: str) -> Optional[float]:
        """
        获取指定指标的最新值
        
        Args:
            metric_name: 指标名称
            
        Returns:
            最新值
        """
        history = self.get_history(metric_name)
        return history[-1] if history else None
    
    def summary(self) -> dict:
        """
        获取所有跟踪指标的摘要
        
        Returns:
            摘要字典
        """
        summary = {}
        for metric_name in self.metrics_to_track:
            if metric_name in self.best_values:
                summary[f'best_{metric_name}'] = self.best_values[metric_name]
                summary[f'best_{metric_name}_epoch'] = self.best_epochs[metric_name]
            
            if metric_name in self.history and self.history[metric_name]:
                summary[f'latest_{metric_name}'] = self.history[metric_name][-1]
        
        return summary
