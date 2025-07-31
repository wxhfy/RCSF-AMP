#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
日志设置工具
提供统一的日志配置和管理功能
"""

import logging
import sys
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
