#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
日志设置工具
提供统一的日志配置和管理功能
"""

import logging
import sys
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, Union


def setup_logging(
    log_file: Optional[Union[str, Path]] = None,
    log_level: Union[str, int] = logging.INFO,
    console_output: bool = True,
    file_output: bool = True,
    formatter_style: str = 'detailed'
) -> logging.Logger:
    """
    设置日志系统
    
    Args:
        log_file: 日志文件路径，如果为None则自动生成
        log_level: 日志级别
        console_output: 是否输出到控制台
        file_output: 是否输出到文件
        formatter_style: 格式化样式 ('simple', 'detailed', 'timestamp')
        
    Returns:
        配置好的logger对象
    """
    # 创建根logger
    logger = logging.getLogger()
    logger.setLevel(log_level)
    
    # 清除现有的handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # 选择格式化器
    if formatter_style == 'simple':
        formatter = logging.Formatter('%(levelname)s: %(message)s')
    elif formatter_style == 'timestamp':
        formatter = logging.Formatter('%(asctime)s - %(message)s')
    else:  # detailed
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    handlers = []
    
    # 控制台输出
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        handlers.append(console_handler)
    
    # 文件输出
    if file_output:
        if log_file is None:
            # 自动生成日志文件名
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_file = f'training_{timestamp}.log'
        
        # 确保日志目录存在
        log_path = Path(log_file)
        # 目录创建由主训练脚本负责，这里不再创建多余目录
        
        file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)
    
    # 添加所有handlers
    for handler in handlers:
        logger.addHandler(handler)
    
    return logger


def get_logger(name: str, level: Union[str, int] = logging.INFO) -> logging.Logger:
    """
    获取指定名称的logger
    
    Args:
        name: logger名称
        level: 日志级别
        
    Returns:
        logger对象
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    return logger


def log_system_info(logger: logging.Logger):
    """
    记录系统信息
    
    Args:
        logger: logger对象
    """
    try:
        import torch
        import platform
        
        logger.info("=== 系统信息 ===")
        logger.info(f"Python版本: {platform.python_version()}")
        logger.info(f"操作系统: {platform.system()} {platform.release()}")
        logger.info(f"PyTorch版本: {torch.__version__}")
        logger.info(f"CUDA可用: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            logger.info(f"CUDA版本: {torch.version.cuda}")
            logger.info(f"GPU数量: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        
        logger.info("=============")
        
    except Exception as e:
        logger.warning(f"记录系统信息时出错: {e}")


def log_model_info(model, logger: logging.Logger):
    """
    记录模型信息
    
    Args:
        model: 模型对象
        logger: logger对象
    """
    try:
        import torch
        
        if hasattr(model, 'parameters'):
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            logger.info("=== 模型信息 ===")
            logger.info(f"总参数量: {total_params:,}")
            logger.info(f"可训练参数量: {trainable_params:,}")
            logger.info(f"冻结参数量: {total_params - trainable_params:,}")
            logger.info(f"模型大小: {total_params * 4 / (1024**2):.2f} MB")
            logger.info("=============")
        
    except Exception as e:
        logger.warning(f"记录模型信息时出错: {e}")


def log_config_info(config: dict, logger: logging.Logger):
    """
    记录配置信息
    
    Args:
        config: 配置字典
        logger: logger对象
    """
    try:
        logger.info("=== 配置信息 ===")
        
        # 递归打印配置
        def print_config(cfg, prefix=""):
            for key, value in cfg.items():
                if isinstance(value, dict):
                    logger.info(f"{prefix}{key}:")
                    print_config(value, prefix + "  ")
                else:
                    logger.info(f"{prefix}{key}: {value}")
        
        print_config(config)
        logger.info("=============")
        
    except Exception as e:
        logger.warning(f"记录配置信息时出错: {e}")


def log_training_start(logger: logging.Logger, config: dict = None):
    """
    记录训练开始信息
    
    Args:
        logger: logger对象
        config: 配置字典
    """
    logger.info("=" * 60)
    logger.info("🚀 开始训练")
    logger.info(f"⏰ 训练开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)
    
    # 记录系统信息
    log_system_info(logger)
    
    # 记录配置信息
    if config:
        log_config_info(config, logger)


def log_training_end(logger: logging.Logger, start_time: datetime = None):
    """
    记录训练结束信息
    
    Args:
        logger: logger对象
        start_time: 训练开始时间
    """
    end_time = datetime.now()
    logger.info("=" * 60)
    logger.info("🎉 训练完成")
    logger.info(f"⏰ 训练结束时间: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    if start_time:
        duration = end_time - start_time
        logger.info(f"⏱️ 总训练时间: {duration}")
    
    logger.info("=" * 60)


def log_epoch_metrics(
    epoch: int,
    train_metrics: dict,
    val_metrics: dict = None,
    logger: logging.Logger = None,
    log_level: int = logging.INFO
):
    """
    记录epoch指标
    
    Args:
        epoch: epoch数
        train_metrics: 训练指标
        val_metrics: 验证指标
        logger: logger对象
        log_level: 日志级别
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    message = f"Epoch {epoch}"
    
    # 训练指标
    if train_metrics:
        train_str = " | ".join([f"{k}: {v:.4f}" for k, v in train_metrics.items() if isinstance(v, (int, float))])
        message += f" | Train: {train_str}"
    
    # 验证指标
    if val_metrics:
        val_str = " | ".join([f"{k}: {v:.4f}" for k, v in val_metrics.items() if isinstance(v, (int, float))])
        message += f" | Val: {val_str}"
    
    logger.log(log_level, message)


class ProgressLogger:
    """
    进度日志记录器
    """
    
    def __init__(self, logger: logging.Logger, total: int, prefix: str = ""):
        self.logger = logger
        self.total = total
        self.prefix = prefix
        self.current = 0
        self.last_percentage = -1
    
    def update(self, step: int = 1):
        """更新进度"""
        self.current += step
        percentage = int(100 * self.current / self.total)
        
        # 每10%记录一次
        if percentage >= self.last_percentage + 10:
            self.logger.info(f"{self.prefix}进度: {percentage}% ({self.current}/{self.total})")
            self.last_percentage = percentage
    
    def finish(self):
        """完成进度"""
        self.logger.info(f"{self.prefix}完成: 100% ({self.total}/{self.total})")


def create_experiment_logger(
    experiment_name: str,
    log_dir: Union[str, Path] = "logs",
    console_output: bool = True
) -> logging.Logger:
    """
    创建实验专用的logger
    
    Args:
        experiment_name: 实验名称
        log_dir: 日志目录
        console_output: 是否输出到控制台
        
    Returns:
        配置好的logger
    """
    # 创建日志目录
    log_dir = Path(log_dir)
    # 目录创建由主训练脚本负责，这里不再创建多余目录
    
    # 生成日志文件名
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f"{experiment_name}_{timestamp}.log"
    
    # 设置logger
    logger = setup_logging(
        log_file=log_file,
        console_output=console_output,
        formatter_style='detailed'
    )
    
    logger.info(f"实验日志已创建: {log_file}")
    return logger
