#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
配置文件工具
用于加载、验证和处理配置文件
"""

import yaml
import json
from pathlib import Path
from typing import Dict, Any, Union, Optional
import logging

logger = logging.getLogger(__name__)


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    加载配置文件
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        配置字典
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            if config_path.suffix.lower() in ['.yaml', '.yml']:
                config = yaml.safe_load(f)
            elif config_path.suffix.lower() == '.json':
                config = json.load(f)
            else:
                raise ValueError(f"不支持的配置文件格式: {config_path.suffix}")
        
        logger.info(f"成功加载配置文件: {config_path}")
        return config
        
    except Exception as e:
        logger.error(f"加载配置文件失败: {e}")
        raise

def validate_config(config: Dict[str, Any]) -> None:
    """
    验证配置文件的完整性和正确性
    
    Args:
        config: 配置字典
        
    Raises:
        ValueError: 配置验证失败
    """
    required_sections = ['model', 'training', 'paths']
    
    for section in required_sections:
        if section not in config:
            raise ValueError(f"配置文件缺少必需的部分: {section}")
    
    # 验证模型配置
    model_config = config['model']
    
    # 检查新的模块化配置格式
    if any(key in model_config for key in ['esm_lora', 'node_encoder', 'edge_encoder']):
        # 新的模块化配置格式 - 检查主要模块是否存在
        expected_modules = ['node_encoder', 'edge_encoder', 'prediction_head']
        for module in expected_modules:
            if module not in model_config:
                raise ValueError(f"模型配置缺少必需的模块: {module}")
        
        # 检查关键配置
        if 'node_encoder' in model_config:
            node_config = model_config['node_encoder']
            if 'hidden_dim' not in node_config:
                raise ValueError(f"node_encoder配置缺少必需的键: hidden_dim")
    
    # 检查架构配置（旧格式兼容）
    elif 'architecture' in model_config:
        arch_config = model_config['architecture']
        required_arch_keys = ['hidden_dim']
        for key in required_arch_keys:
            if key not in arch_config:
                raise ValueError(f"模型架构配置缺少必需的键: {key}")
    else:
        # 向后兼容：检查是否直接在model下（旧格式）
        required_model_keys = ['hidden_dim']
        for key in required_model_keys:
            if key not in model_config:
                # 尝试在其他地方找到hidden_dim
                found_hidden_dim = False
                for module_name, module_config in model_config.items():
                    if isinstance(module_config, dict) and 'hidden_dim' in module_config:
                        found_hidden_dim = True
                        break
                
                if not found_hidden_dim:
                    raise ValueError(f"模型配置缺少必需的键: {key}，请在任一模块配置中提供")
    
    # 验证训练配置
    training_config = config['training']
    
    # 检查子阶段配置
    if 'sub_stages' in training_config:
        for stage_name, stage_config in training_config['sub_stages'].items():
            # 检查优化器配置（新格式）
            if 'optimizer' in stage_config:
                optimizer_config = stage_config['optimizer']
                if 'lr' not in optimizer_config:
                    raise ValueError(f"训练阶段 {stage_name} 的优化器配置缺少学习率(lr)")
            # 兼容旧格式
            elif 'learning_rates' not in stage_config:
                raise ValueError(f"训练阶段 {stage_name} 缺少学习率配置(optimizer.lr 或 learning_rates)")
    else:
        # 单阶段训练需要直接的learning_rate
        if 'learning_rate' not in training_config and 'optimizer' not in training_config:
            raise ValueError(f"训练配置缺少学习率配置(learning_rate 或 optimizer)")
        
        # 如果有optimizer配置，检查其中的lr
        if 'optimizer' in training_config:
            optimizer_config = training_config['optimizer']
            if 'lr' not in optimizer_config:
                raise ValueError(f"训练配置的优化器缺少学习率(lr)")
        
        if 'learning_rate' not in training_config:
            raise ValueError(f"训练配置缺少必需的键: learning_rate")
    
    # 验证路径配置
    paths_config = config['paths']
    required_path_keys = ['data_root']
    
    for key in required_path_keys:
        if key not in paths_config:
            raise ValueError(f"路径配置缺少必需的键: {key}")
    
    logger.info("配置文件验证通过")

def get_evaluation_threshold(config: Dict[str, Any], default: float = 0.550) -> float:
    """
    从配置中获取评估阈值
    
    Args:
        config: 配置字典
        default: 默认阈值
        
    Returns:
        评估阈值
    """
    try:
        threshold = config.get('training', {}).get('evaluation', {}).get('threshold', default)
        logger.info(f"使用评估阈值: {threshold}")
        return float(threshold)
    except (KeyError, ValueError, TypeError) as e:
        logger.warning(f"无法从配置中获取阈值，使用默认值 {default}: {e}")
        return default


    """
    从配置中获取阈值优化设置
    
    Args:
        config: 配置字典
        
    Returns:
        阈值优化配置字典
    """
    default_config = {
        'enabled': True,
        'metric': 'mcc',
        'min_threshold': 0.01,
        'max_threshold': 0.99,
        'num_thresholds': 99
    }
    
    try:
        threshold_opt_config = config.get('training', {}).get('evaluation', {}).get('threshold_optimization', {})
        # 合并默认配置和用户配置
        result = {**default_config, **threshold_opt_config}
        logger.info(f"阈值优化配置: {result}")
        return result
    except Exception as e:
        logger.warning(f"无法从配置中获取阈值优化设置，使用默认配置: {e}")
        return default_config
