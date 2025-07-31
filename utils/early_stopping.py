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
