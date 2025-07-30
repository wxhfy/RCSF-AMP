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


def save_config(config: Dict[str, Any], save_path: Union[str, Path]) -> None:
    """
    保存配置文件
    
    Args:
        config: 配置字典
        save_path: 保存路径
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(save_path, 'w', encoding='utf-8') as f:
            if save_path.suffix.lower() in ['.yaml', '.yml']:
                yaml.dump(config, f, default_flow_style=False, allow_unicode=True, indent=2)
            elif save_path.suffix.lower() == '.json':
                json.dump(config, f, indent=2, ensure_ascii=False)
            else:
                raise ValueError(f"不支持的配置文件格式: {save_path.suffix}")
        
        logger.info(f"成功保存配置文件: {save_path}")
        
    except Exception as e:
        logger.error(f"保存配置文件失败: {e}")
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


def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    合并配置字典，override_config会覆盖base_config中的相同键
    
    Args:
        base_config: 基础配置
        override_config: 覆盖配置
        
    Returns:
        合并后的配置
    """
    import copy
    
    merged = copy.deepcopy(base_config)
    
    def recursive_update(d, u):
        for k, v in u.items():
            if isinstance(v, dict):
                d[k] = recursive_update(d.get(k, {}), v)
            else:
                d[k] = v
        return d
    
    return recursive_update(merged, override_config)


def get_config_value(config: Dict[str, Any], key_path: str, default: Any = None) -> Any:
    """
    通过点分隔的路径获取配置值
    
    Args:
        config: 配置字典
        key_path: 键路径，如 'model.hidden_dim'
        default: 默认值
        
    Returns:
        配置值
    """
    keys = key_path.split('.')
    value = config
    
    try:
        for key in keys:
            value = value[key]
        return value
    except (KeyError, TypeError):
        return default


def set_config_value(config: Dict[str, Any], key_path: str, value: Any) -> None:
    """
    通过点分隔的路径设置配置值
    
    Args:
        config: 配置字典
        key_path: 键路径，如 'model.hidden_dim'
        value: 要设置的值
    """
    keys = key_path.split('.')
    current = config
    
    for key in keys[:-1]:
        if key not in current:
            current[key] = {}
        current = current[key]
    
    current[keys[-1]] = value


def create_default_config() -> Dict[str, Any]:
    """
    创建默认配置
    
    Returns:
        默认配置字典
    """
    return {
        'model': {
            'hidden_dim': 256,
            'num_heads': 8,
            'dropout': 0.1,
            'use_mamba': True,
            'mamba_config': {
                'd_state': 16,
                'd_conv': 4,
                'expand': 2
            }
        },
        'training': {
            'batch_size': 32,
            'learning_rate': 1e-4,
            'num_epochs': 100,
            'optimizer': 'adamw',
            'scheduler': {
                'type': 'cosine_annealing',
                'warmup_epochs': 5,
                'T_max': 100,
                'min_lr_factor': 0.01
            },
            'loss_config': {
                'alignment_contrastive_temperature': 0.1,
                'supervised_contrastive_temperature': 0.07,
                'label_smoothing': 0.1
            }
        },
        'paths': {
            'data_root': '/path/to/data',
            'checkpoint_dir': 'checkpoints',
            'log_dir': 'logs'
        },
        'logging': {
            'level': 'INFO',
            'console_output': True,
            'file_output': True
        }
    }


def validate_paths(config: Dict[str, Any]) -> None:
    """
    验证配置中的路径是否存在
    
    Args:
        config: 配置字典
    """
    paths_to_check = []
    
    # 检查数据路径
    if 'paths' in config and 'data_root' in config['paths']:
        data_root = Path(config['paths']['data_root'])
        if not data_root.exists():
            logger.warning(f"数据根目录不存在: {data_root}")
        else:
            paths_to_check.append(data_root)
    
    # 创建输出目录
    output_dirs = []
    if 'paths' in config:
        paths_config = config['paths']
        
        if 'checkpoint_dir' in paths_config:
            output_dirs.append(Path(paths_config['checkpoint_dir']))
        
        if 'log_dir' in paths_config:
            output_dirs.append(Path(paths_config['log_dir']))
    
    for dir_path in output_dirs:
        try:
            dir_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"创建输出目录: {dir_path}")
        except Exception as e:
            logger.warning(f"无法创建目录 {dir_path}: {e}")


def get_stage_config(config: Dict[str, Any], stage: str) -> Dict[str, Any]:
    """
    获取特定阶段的配置
    
    Args:
        config: 完整配置
        stage: 阶段名称 (如 '1.0', '2.0')
        
    Returns:
        该阶段的配置
    """
    if 'training' not in config or 'sub_stages' not in config['training']:
        return {}
    
    sub_stages = config['training']['sub_stages']
    if stage not in sub_stages:
        logger.warning(f"未找到阶段配置: {stage}")
        return {}
    
    return sub_stages[stage]


def update_config_for_stage(config: Dict[str, Any], stage: str) -> Dict[str, Any]:
    """
    根据阶段更新配置
    
    Args:
        config: 基础配置
        stage: 阶段名称
        
    Returns:
        更新后的配置
    """
    import copy
    
    updated_config = copy.deepcopy(config)
    stage_config = get_stage_config(config, stage)
    
    if stage_config:
        # 更新学习率
        if 'learning_rates' in stage_config:
            lr_config = stage_config['learning_rates']
            if 'default' in lr_config:
                updated_config['training']['learning_rate'] = lr_config['default']
        
        # 更新epoch数
        if 'epochs' in stage_config:
            updated_config['training']['num_epochs'] = stage_config['epochs']
        
        # 更新损失权重
        if 'loss_weights' in stage_config:
            if 'loss_config' not in updated_config['training']:
                updated_config['training']['loss_config'] = {}
            updated_config['training']['loss_config']['loss_weights'] = stage_config['loss_weights']
        
        # 更新激活的损失
        if 'active_losses' in stage_config:
            if 'loss_config' not in updated_config['training']:
                updated_config['training']['loss_config'] = {}
            updated_config['training']['loss_config']['active_losses'] = stage_config['active_losses']
        
        # 更新早停配置
        if 'early_stopping' in stage_config:
            updated_config['training']['early_stopping'] = stage_config['early_stopping']
    
    return updated_config


def print_config_summary(config: Dict[str, Any], logger_obj: Optional[logging.Logger] = None) -> None:
    """
    打印配置摘要
    
    Args:
        config: 配置字典
        logger_obj: 日志对象
    """
    if logger_obj is None:
        logger_obj = logger
    
    logger_obj.info("=== 配置摘要 ===")
    
    # 模型配置
    if 'model' in config:
        model_config = config['model']
        logger_obj.info(f"模型维度: {model_config.get('hidden_dim', 'N/A')}")
        logger_obj.info(f"注意力头数: {model_config.get('num_heads', 'N/A')}")
        logger_obj.info(f"Dropout: {model_config.get('dropout', 'N/A')}")
        logger_obj.info(f"使用Mamba: {model_config.get('use_mamba', False)}")
    
    # 训练配置
    if 'training' in config:
        training_config = config['training']
        logger_obj.info(f"批大小: {training_config.get('batch_size', 'N/A')}")
        logger_obj.info(f"学习率: {training_config.get('learning_rate', 'N/A')}")
        logger_obj.info(f"训练轮数: {training_config.get('num_epochs', 'N/A')}")
        logger_obj.info(f"优化器: {training_config.get('optimizer', 'N/A')}")
    
    # 路径配置
    if 'paths' in config:
        paths_config = config['paths']
        logger_obj.info(f"数据路径: {paths_config.get('data_root', 'N/A')}")
        logger_obj.info(f"检查点目录: {paths_config.get('checkpoint_dir', 'N/A')}")
        logger_obj.info(f"日志目录: {paths_config.get('log_dir', 'N/A')}")
    
    logger_obj.info("=" * 20)


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


def get_threshold_optimization_config(config: Dict[str, Any]) -> Dict[str, Any]:
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
