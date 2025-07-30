#!/usr/bin/env python3
"""
SUCF-AMP训练脚本
支持两阶段训练：模态对齐预热 + 协同融合与预测
"""

import os
import sys
import argparse
import logging
import yaml
import time
import numpy as np
import torch
from pathlib import Path
from datetime import datetime, timedelta
from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch_geometric.data import Data
from tqdm import tqdm

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.sgg_net_model import create_sucf_model
from training.sucf_losses import create_sucf_loss_function
from utils.metrics import calculate_metrics
from utils.early_stopping import EarlyStopping
from utils.logging_utils import setup_logging
from utils.config_utils import load_config, validate_config, get_evaluation_threshold


# 设置日志
logger = logging.getLogger(__name__)


class SUCFTrainer:
    """SUCF训练器，支持两阶段训练"""

    def __init__(self, config_path):
        """初始化训练器"""
        self.is_main_process = True
        
        # 加载配置
        self.config = load_config(config_path)
        validate_config(self.config)
        
        # 评估配置
        self.eval_threshold = get_evaluation_threshold(self.config)
        
        # 设置设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"使用设备: {self.device}")
        
        # 创建输出目录（所有进程都需要知道路径）
        self._create_output_dirs()
        
        # 初始化模型
        self.model = self._create_model()
        
        # 不使用DDP，直接保存模型
        self.model_without_ddp = self.model
        
        # 初始化损失函数
        self.loss_fn = create_sucf_loss_function(self.config)
        
        # 初始化优化器和调度器
        self.optimizer = None
        self.scheduler = None
        
        # 训练状态
        self.current_stage = None
        self.current_epoch = 0
        self.best_metrics = {}
        
        # 设置日志文件写入log_dir
        log_file_path = self.log_dir / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.handlers.clear()
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)
        logger.setLevel(logging.INFO)
        logger.info(f"日志文件写入: {log_file_path}")
        logger.info("SUCF训练器初始化完成")

    # 分布式相关已移除，无需barrier
    
    def _create_output_dirs(self):
        """创建输出目录"""
        paths = self.config.get('paths', {})
        
        self.checkpoint_dir = Path(paths.get('checkpoint_dir', './best_structure_config0/checkpoints'))
        self.log_dir = Path(paths.get('log_dir', './best_structure_config0/logs'))
        
        # 只有主进程创建目录
        # 只在指定目录下创建日志和检查点
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"检查点目录: {self.checkpoint_dir}")
        logger.info(f"日志目录: {self.log_dir}")
    
    def _create_model(self):
        """创建模型"""
        model_config = self.config.get('model', {})
        model = create_sucf_model(model_config)
        model = model.to(self.device)
        
        # 打印模型信息
        model_info = model.get_model_info()
        logger.info(f"模型参数总数: {model_info['total_params']:,}")
        logger.info(f"可训练参数: {model_info['trainable_params']:,}")
        
        return model
    
    def _create_optimizer_and_scheduler(self, stage_config):
        """为当前阶段创建优化器和调度器"""
        logger.info(f"开始为阶段 {self.current_stage} 创建优化器和调度器")
        
        training_config = self.config.get('training', {})
        optimizer_config = training_config.get('optimizer_params', {})
        scheduler_config = training_config.get('scheduler_params', {})
        
        # 获取学习率 - 从stage_config的optimizer配置中读取
        optimizer_stage_config = stage_config.get('optimizer', {})
        default_lr = float(optimizer_stage_config.get('lr', 1e-3))  
        
        # 检查是否有自定义学习率配置
        learning_rates = stage_config.get('learning_rates', {})
        if learning_rates:
            default_lr = float(learning_rates.get('default', default_lr))
            head_lr = float(learning_rates.get('head_params', default_lr))
        else:
            head_lr = default_lr
        
        # 分层学习率 
        param_groups = []
        
        # 预测头参数使用更高学习率
        head_params = []
        for name, param in self.model_without_ddp.named_parameters():
            if 'activity_predictor' in name or 'global_pooling' in name:
                head_params.append(param)
        
        if head_params:
            param_groups.append({
                'params': head_params,
                'lr': head_lr,
                'name': 'head_params'
            })
        
        # 其他参数使用默认学习率
        other_params = []
        head_param_ids = set(id(p) for p in head_params)
        for param in self.model_without_ddp.parameters():
            if id(param) not in head_param_ids:
                other_params.append(param)
        
        if other_params:
            param_groups.append({
                'params': other_params,
                'lr': default_lr,
                'name': 'default'
            })
        
        # 创建优化器
        optimizer_type = optimizer_config.get('type', 'AdamW')
        if optimizer_type == 'AdamW':
            self.optimizer = torch.optim.AdamW(
                param_groups,
                weight_decay=optimizer_config.get('weight_decay', 0.01),
                betas=optimizer_config.get('betas', [0.9, 0.999]),
                eps=optimizer_config.get('eps', 1e-8)
            )
        else:
            raise ValueError(f"不支持的优化器类型: {optimizer_type}")
        
        # 创建调度器
        scheduler_type = scheduler_config.get('scheduler_type', 'cosine_annealing')
        if scheduler_type == 'cosine_annealing':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=scheduler_config.get('T_max', 100),
                eta_min=default_lr * scheduler_config.get('min_lr_factor', 0.01)
            )
        else:
            self.scheduler = None
        
        logger.info(f"创建优化器: {optimizer_type}, 参数组数: {len(param_groups)}")
        logger.info(f"学习率 - 默认: {default_lr}, 预测头: {head_lr}")
    
    def train_epoch(self, train_loader, stage_config):
        """训练一个epoch"""
        self.model.train()
        
        total_loss = 0.0
        total_samples = 0
        epoch_metrics = {}
        
        # 获取阶段信息
        stage_info = {
            'active_losses': stage_config.get('active_losses', ['activity']),
            'loss_weights': stage_config.get('loss_weights', {'activity': 1.0})
        }
        
        # 只在主进程显示进度条
        pbar = tqdm(train_loader, desc='Train', ncols=80)
        
        for batch_idx, batch in enumerate(pbar):
            batch = batch.to(self.device)
            
            # 前向传播
            outputs = self.model(batch)
            
            # 计算损失
            losses = self.loss_fn(outputs, batch, stage_info)
            total_loss_value = losses['total_loss']
            
            # 反向传播
            self.optimizer.zero_grad()
            total_loss_value.backward()
            
            # 梯度裁剪
            gradient_clip_norm = self.config.get('common_training', {}).get('gradient_clip_norm', 1.0)
            if gradient_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), gradient_clip_norm)
            
            self.optimizer.step()
            
            # 累积统计
            batch_size = batch.y.size(0) if hasattr(batch, 'y') and batch.y is not None else len(batch)
            total_loss += total_loss_value.item() * batch_size
            total_samples += batch_size
            
            # 更新进度条（仅主进程）
            pbar.set_postfix({
                'loss': f"{total_loss_value.item():.4f}",
                'avg_loss': f"{total_loss/total_samples:.4f}"
            })
        
        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
        
        if self.scheduler:
            self.scheduler.step()
        
        epoch_metrics = {
            'train_loss': avg_loss,
            'train_samples': total_samples
        }
        
        return epoch_metrics
    
    def validate_epoch(self, val_loader, stage_config):
        """验证一个epoch"""
        self.model.eval()
        
        total_loss = 0.0
        total_samples = 0
        all_predictions = []
        all_targets = []
        epoch_metrics = {}
        
        # 获取阶段信息
        stage_info = {
            'active_losses': stage_config.get('active_losses', ['activity']),
            'loss_weights': stage_config.get('loss_weights', {'activity': 1.0})
        }
        
        # 只在主进程显示进度条
        pbar = tqdm(val_loader, desc='Val', ncols=80)
        
        with torch.no_grad():
            for batch in pbar:
                batch = batch.to(self.device)
                
                # 前向传播
                outputs = self.model(batch)
                
                # 计算损失
                losses = self.loss_fn(outputs, batch, stage_info)
                total_loss_value = losses['total_loss']
                
                # 收集预测和目标
                if 'activity_pred' in outputs:
                    batch_predictions = torch.sigmoid(outputs['activity_pred']).cpu().numpy()
                    batch_targets = batch.y.cpu().numpy()
                    
                    all_predictions.extend(batch_predictions)
                    all_targets.extend(batch_targets)
                
                # 累积统计
                batch_size = batch.y.size(0) if hasattr(batch, 'y') and batch.y is not None else len(batch)
                total_loss += total_loss_value.item() * batch_size
                total_samples += batch_size
                
                # 更新进度条（仅主进程）
                pbar.set_postfix({'loss': f"{total_loss_value.item():.4f}"})
        
        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
        
        # 计算指标（仅主进程或非分布式）
        val_metrics = {'val_loss': avg_loss}
        if len(all_predictions) > 0 and len(all_targets) > 0:
            try:
                # 使用配置中的优化阈值进行二分类
                metrics = calculate_metrics(
                    np.array(all_targets), 
                    np.array(all_predictions),  # 传入概率
                    threshold=self.eval_threshold  # 使用配置中的阈值
                )
                val_metrics.update({f'val_{k}': v for k, v in metrics.items()})
                
                # 详细的验证日志
                positive_ratio = np.mean(all_targets)
                pred_positive_ratio = np.mean(np.array(all_predictions) > self.eval_threshold)
                
                logger.info(f"验证集标签分布 - 正样本比例: {positive_ratio:.3f}")
                logger.info(f"验证集预测分布 - 预测正样本比例: {pred_positive_ratio:.3f}")
                logger.info(f"验证指标: {self._format_metrics(metrics)}")
            
            except Exception as e:
                logger.warning(f"验证指标计算失败: {e}")
        
        return val_metrics
    
    def optimize_threshold_on_validation(self, val_loader, stage_config, all_predictions=None, all_targets=None):
        """在验证集上优化二分类阈值"""
        # 获取当前阶段名称
        current_stage = self.current_stage
        
        # 只在第二阶段（分类微调阶段）执行阈值优化
        # 第一阶段是特征对齐阶段，不涉及分类任务
        if current_stage != "2.0":
            if self.is_main_process:
                logger.debug(f"跳过阶段 {current_stage} 的阈值优化（仅在阶段2.0执行）")
            return self.eval_threshold, {}
        
        # 获取动态阈值配置
        dynamic_threshold_config = self.config.get('training', {}).get('evaluation', {}).get('dynamic_threshold', {})
        
        if not dynamic_threshold_config.get('enabled', False):
            return self.eval_threshold, {}
        
        # 如果没有传入预测结果，则重新运行验证
        if all_predictions is None or all_targets is None:
            self.model.eval()
            all_predictions = []
            all_targets = []
            
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(self.device)
                    outputs = self.model(batch)
                    
                    if 'activity_pred' in outputs:
                        batch_predictions = torch.sigmoid(outputs['activity_pred']).cpu().numpy()
                        batch_targets = batch.y.cpu().numpy()
                        
                        all_predictions.extend(batch_predictions)
                        all_targets.extend(batch_targets)
            
            # 在分布式环境中同步结果
            if self.is_distributed:
                all_predictions_np = np.array(all_predictions)
                all_targets_np = np.array(all_targets)
                
                all_predictions_tensor = torch.from_numpy(all_predictions_np).to(self.device)
                all_targets_tensor = torch.from_numpy(all_targets_np).to(self.device)
                
                gathered_predictions = [torch.zeros_like(all_predictions_tensor) for _ in range(self.world_size)]
                gathered_targets = [torch.zeros_like(all_targets_tensor) for _ in range(self.world_size)]
                
                
                if self.is_main_process:
                    all_predictions = torch.cat(gathered_predictions).cpu().numpy()
                    all_targets = torch.cat(gathered_targets).cpu().numpy()
        
        # 只在主进程执行阈值搜索
        if not self.is_main_process:
            return self.eval_threshold, {}
        
        # 获取搜索参数
        search_range = dynamic_threshold_config.get('search_range', {})
        min_threshold = search_range.get('min_threshold', 0.3)
        max_threshold = search_range.get('max_threshold', 0.95)
        search_steps = search_range.get('search_steps', 50)
        optimization_metric = dynamic_threshold_config.get('optimization_metric', 'mcc')
        log_details = dynamic_threshold_config.get('log_search_details', False)
        
        # 生成阈值搜索空间
        thresholds = np.linspace(min_threshold, max_threshold, search_steps)
        
        best_threshold = self.eval_threshold
        best_score = -float('inf')
        threshold_results = []
        
        logger.info(f"开始动态阈值搜索: 范围[{min_threshold:.3f}, {max_threshold:.3f}], 步数={search_steps}, 优化指标={optimization_metric}")
        
        # 搜索最优阈值
        for threshold in thresholds:
            try:
                metrics = calculate_metrics(
                    np.array(all_targets), 
                    np.array(all_predictions),
                    threshold=threshold
                )
                
                # 获取目标指标
                score = metrics.get(optimization_metric, 0)
                threshold_results.append({
                    'threshold': threshold,
                    'score': score,
                    'metrics': metrics
                })
                
                if score > best_score:
                    best_score = score
                    best_threshold = threshold
                    
            except Exception as e:
                logger.warning(f"阈值 {threshold:.3f} 计算失败: {e}")
                continue
        
        # 记录搜索结果
        search_summary = {
            'best_threshold': best_threshold,
            'best_score': best_score,
            'optimization_metric': optimization_metric,
            'search_range': f"[{min_threshold:.3f}, {max_threshold:.3f}]",
            'search_steps': search_steps,
            'threshold_change': abs(best_threshold - self.eval_threshold)
        }
        
        logger.info(f"动态阈值搜索完成:")
        logger.info(f"  最优阈值: {best_threshold:.4f} (原阈值: {self.eval_threshold:.4f})")
        logger.info(f"  最优{optimization_metric}: {best_score:.4f}")
        logger.info(f"  阈值变化: {search_summary['threshold_change']:.4f}")
        
        if log_details and len(threshold_results) > 0:
            # 显示前5个最佳阈值
            sorted_results = sorted(threshold_results, key=lambda x: x['score'], reverse=True)[:5]
            logger.info("前5个最佳阈值:")
            for i, result in enumerate(sorted_results):
                metrics_str = self._format_metrics(result['metrics'])
                logger.info(f"  {i+1}. 阈值={result['threshold']:.4f}, {optimization_metric}={result['score']:.4f}, {metrics_str}")
        
        return best_threshold, search_summary
    
    def should_update_dynamic_threshold(self, epoch, stage_name):
        """判断是否应该更新动态阈值"""
        # 只在第二阶段（分类微调阶段）启用阈值优化
        # 第一阶段是特征对齐阶段，不涉及分类任务，无需阈值优化
        if stage_name != "2.0":
            return False
            
        dynamic_threshold_config = self.config.get('training', {}).get('evaluation', {}).get('dynamic_threshold', {})
        
        if not dynamic_threshold_config.get('enabled', False):
            return False
        
        # 检查是否达到开始epoch
        start_epoch = dynamic_threshold_config.get('start_epoch', 10)
        if epoch < start_epoch:
            return False
        
        # 检查更新频率
        update_frequency = dynamic_threshold_config.get('update_frequency', 5)
        if (epoch - start_epoch) % update_frequency != 0:
            return False
        
        return True
    
    def update_eval_threshold(self, new_threshold):
        """更新评估阈值"""
        old_threshold = self.eval_threshold
        self.eval_threshold = new_threshold
        
        logger.info(f"评估阈值已更新: {old_threshold:.4f} -> {new_threshold:.4f}")
        
        return old_threshold

    def save_checkpoint(self, checkpoint_name, metrics=None, is_best=False):
        """保存检查点"""
        checkpoint = {
            'model_state_dict': self.model_without_ddp.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'current_stage': self.current_stage,
            'current_epoch': self.current_epoch,
            'eval_threshold': self.eval_threshold,  # 保存当前评估阈值
            'metrics': metrics or {},
            'config': self.config
        }
        
        checkpoint_path = self.checkpoint_dir / f"{checkpoint_name}.pth"
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"保存检查点: {checkpoint_path}")
        
        if is_best:
            best_path = self.checkpoint_dir / f"best_{self.current_stage}.pth"
            torch.save(checkpoint, best_path)
            logger.info(f"保存最优模型: {best_path}")
    
    def load_checkpoint(self, checkpoint_path, load_optimizer=False):
        """加载检查点，所有进程执行加载，但只有主进程打印日志"""
        if not os.path.exists(checkpoint_path):
            logger.warning(f"检查点文件不存在: {checkpoint_path}，跳过加载")
            return False
        
        logger.info(f"加载检查点: {checkpoint_path}")
            
        # 所有进程都从磁盘加载，确保map_location正确设置到每个进程的设备上
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # 所有进程都加载模型状态
        self.model_without_ddp.load_state_dict(checkpoint['model_state_dict'])
        
        # 优化器和调度器状态通常也需要所有进程加载
        if load_optimizer and self.optimizer and 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if load_optimizer and self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # 更新训练元数据（所有进程都需要知道）
        self.current_stage = checkpoint.get('current_stage', self.current_stage)
        self.current_epoch = checkpoint.get('current_epoch', 0)
        
        # 恢复评估阈值
        if 'eval_threshold' in checkpoint:
            self.eval_threshold = checkpoint['eval_threshold']
            logger.info(f"恢复评估阈值: {self.eval_threshold:.4f}")
        
        logger.info(f"成功加载检查点，阶段: {self.current_stage}, epoch: {self.current_epoch}")
            
        return True
    
    def train_stage(self, stage_name, stage_config, train_loader, val_loader):
        """训练一个阶段 - 简化版本，移除复杂的早停同步逻辑"""
        # 更新当前阶段信息
        self.current_stage = stage_name
        
        logger.info(f"开始训练阶段: {stage_name}")
        logger.info(f"阶段描述: {stage_config.get('description', '')}")
        logger.info(f"计划epochs: {stage_config.get('epochs', 0)}")
        logger.info(f"激活损失: {stage_config.get('active_losses', [])}")
        logger.info(f"损失权重: {stage_config.get('loss_weights', {})}")
        
        # 为当前阶段创建优化器和调度器
        self._create_optimizer_and_scheduler(stage_config)
        
        # 早停配置 - 只在主进程使用
        early_stopping_config = stage_config.get('early_stopping', {})
        early_stopping = None
        monitor_key = early_stopping_config.get('monitor', 'val_loss')
        
        if early_stopping_config.get('enable', False) and self.is_main_process:
            early_stopping = EarlyStopping(
                patience=early_stopping_config.get('patience', 10),
                min_delta=early_stopping_config.get('min_delta', 0.001),
                mode=early_stopping_config.get('mode', 'min'),
                restore_best_weights=False  # 关键：不在早停中恢复权重
            )
        
        # 训练循环
        stage_epochs = stage_config.get('epochs', 10)
        best_stage_metric = float('inf') if early_stopping_config.get('mode', 'min') == 'min' else float('-inf')
        
        # 所有进程都执行完整的epoch循环
        for epoch in range(stage_epochs):
            # 更新当前epoch信息
            self.current_epoch = epoch + 1
            
            epoch_start_time = time.time()
            
            # 训练
            train_metrics = self.train_epoch(train_loader, stage_config)
            
            # 验证
            val_metrics = self.validate_epoch(val_loader, stage_config)
            
            # 动态阈值更新（仅主进程计算，然后同步到所有进程）
            threshold_update_summary = {}
            new_threshold = self.eval_threshold  # 默认保持当前阈值
            
            if self.should_update_dynamic_threshold(epoch + 1, stage_name):
                if self.is_main_process:
                    logger.info(f"触发动态阈值更新检查 (Epoch {epoch+1})")
                    try:
                        # 重新获取验证集预测结果进行阈值优化
                        new_threshold, search_summary = self.optimize_threshold_on_validation(val_loader, stage_config)
                        threshold_update_summary = search_summary
                        threshold_update_summary['old_threshold'] = self.eval_threshold
                        
                    except Exception as e:
                        logger.error(f"动态阈值更新失败: {e}")
                        import traceback
                        logger.error(traceback.format_exc())
                        new_threshold = self.eval_threshold  # 保持原阈值
                
                # 所有进程都更新阈值
                if new_threshold != self.eval_threshold:
                    old_threshold = self.update_eval_threshold(new_threshold)
                    
                    # 使用新阈值重新计算验证指标
                    if self.is_main_process:
                        logger.info("使用新阈值重新计算验证指标...")
                    updated_val_metrics = self.validate_epoch(val_loader, stage_config)
                    val_metrics.update(updated_val_metrics)
            
            epoch_time = time.time() - epoch_start_time
            
            # 记录日志
            if self.is_main_process:
                logger.info(f"阶段 {stage_name}, Epoch {epoch+1}/{stage_epochs}")
                logger.info(f"  训练损失: {train_metrics.get('train_loss', 0):.4f}, 验证损失: {val_metrics.get('val_loss', 0):.4f}")
                logger.info(f"  验证指标: {self._format_metrics(val_metrics)}")
                logger.info(f"  当前阈值: {self.eval_threshold:.4f}")
                
                # 如果有阈值更新信息，记录详细日志
                if threshold_update_summary:
                    logger.info(f"  阈值更新摘要:")
                    logger.info(f"    优化指标: {threshold_update_summary.get('optimization_metric', 'N/A')}")
                    logger.info(f"    最优分数: {threshold_update_summary.get('best_score', 0):.4f}")
                    logger.info(f"    阈值变化: {threshold_update_summary.get('old_threshold', 0):.4f} -> {threshold_update_summary.get('best_threshold', 0):.4f}")
                
                logger.info(f"  耗时: {epoch_time:.2f}s")
            
            # 早停检查 - 只在主进程执行，不影响其他进程
            should_stop = False
            if early_stopping is not None and self.is_main_process:
                # 选择监控指标
                if monitor_key.startswith('val_'):
                    monitor_metric = val_metrics.get(monitor_key, val_metrics.get('val_loss', 0))
                else:
                    monitor_metric = val_metrics.get(f'val_{monitor_key}', val_metrics.get('val_loss', 0))
                
                # 早停判断
                should_stop = early_stopping(monitor_metric, None, epoch+1)  # 不传入model避免权重恢复
                
                # 如果是最佳模型，保存检查点
                if early_stopping.best_score == monitor_metric:
                    best_stage_metric = monitor_metric
                    self.save_checkpoint(f"best_{stage_name}", val_metrics, is_best=True)
                    logger.info(f"保存最佳模型，指标: {monitor_metric:.4f}")
            else:
                # 没有早停时，使用验证损失作为指标
                if self.is_main_process:
                    current_metric = val_metrics.get('val_loss', 0)
                    if current_metric < best_stage_metric:
                        best_stage_metric = current_metric
                        self.save_checkpoint(f"best_{stage_name}", val_metrics, is_best=True)
            
            # 定期保存检查点（仅主进程）
            if self.is_main_process and (epoch + 1) % self.config.get('common_training', {}).get('save_interval', 5) == 0:
                self.save_checkpoint(f"{stage_name}_epoch_{epoch+1}", val_metrics)
            
            # 如果早停被触发，跳出训练循环
            if should_stop:
                if self.is_main_process:
                    logger.info(f"早停触发，在epoch {epoch+1}停止阶段 {stage_name} 的训练")
                break

        
        if self.is_main_process:
            logger.info(f"阶段 {stage_name} 训练完成")
            logger.info(f"最佳指标: {best_stage_metric:.4f}")
        
        return best_stage_metric
    
    def train(self, train_loader, val_loader, test_loader=None):
        """完整的两阶段训练流程 - 简化版本"""
        if self.is_main_process:
            logger.info("开始SGG-Net两阶段训练")
        
        # 获取子阶段配置
        sub_stages = self.config.get('training', {}).get('sub_stages', {})
        
        
        # 按阶段顺序训练
        for stage_name in sorted(sub_stages.keys()):
            stage_config = sub_stages[stage_name]
            if self.is_main_process:
                logger.info(f"准备开始阶段 {stage_name}: {stage_config.get('name', 'Unknown')}")
            if stage_name == "2.0":
                best_model_path_stage1 = self.checkpoint_dir / "best_1.0.pth"
                if self.is_main_process:
                    logger.info("进入第二阶段，准备加载第一阶段最佳模型")
                if best_model_path_stage1.exists():
                    if self.is_main_process:
                        logger.info(f"加载第一阶段检查点: {best_model_path_stage1}")
                    self.load_checkpoint(str(best_model_path_stage1), load_optimizer=False)
                    if self.is_main_process:
                        logger.info("第一阶段模型加载完成")
                else:
                    if self.is_main_process:
                        logger.warning("第一阶段最佳模型不存在，继续使用当前模型权重")
            if self.is_main_process:
                logger.info(f"开始训练阶段 {stage_name}")
            # 训练当前阶段
            stage_metric = self.train_stage(stage_name, stage_config, train_loader, val_loader)
            self.best_metrics[stage_name] = stage_metric
            if self.is_main_process:
                logger.info(f"阶段 {stage_name} 完成，最佳指标: {stage_metric:.4f}")
       
        # 训练完成后，评估测试集或输出训练完毕
        if test_loader is not None and self.is_main_process:
            logger.info("检测到测试集，开始测试集评估...")
            self.evaluate_test_set(test_loader)
        elif self.is_main_process:
            logger.info("训练已完成，无测试集可评估。")
        if self.is_main_process:
            logger.info(f"各阶段最佳指标: {self.best_metrics}")
    
    def evaluate_test_set(self, test_loader):
        """评估测试集"""
        logger.info("开始测试集评估")
        
        # 加载最后阶段的最佳模型
        last_stage = max(self.best_metrics.keys()) if self.best_metrics else "2.0"
        best_model_path = self.checkpoint_dir / f"best_{last_stage}.pth"
        
        if best_model_path.exists():
            logger.info(f"加载最佳模型: {best_model_path}")
            self.load_checkpoint(str(best_model_path), load_optimizer=False)
        else:
            logger.warning("最佳模型不存在，使用当前模型权重进行测试")
        
        # 设置为评估模式
        self.model.eval()
        
        all_predictions = []
        all_targets = []
        total_loss = 0.0
        total_samples = 0
        
        # 使用最后阶段的配置进行测试
        test_stage_config = {
            'active_losses': ['activity'],
            'loss_weights': {'activity': 1.0}
        }
        
        stage_info = {
            'active_losses': test_stage_config['active_losses'],
            'loss_weights': test_stage_config['loss_weights']
        }
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc='Test', ncols=80):
                batch = batch.to(self.device)
                
                # 前向传播
                outputs = self.model(batch)
                
                # 计算损失
                losses = self.loss_fn(outputs, batch, stage_info)
                total_loss_value = losses['total_loss']
                
                # 收集预测和目标
                if 'activity_pred' in outputs:
                    batch_predictions = torch.sigmoid(outputs['activity_pred']).cpu().numpy()
                    batch_targets = batch.y.cpu().numpy()
                    
                    all_predictions.extend(batch_predictions)
                    all_targets.extend(batch_targets)
                
                # 累积统计
                batch_size = batch.y.size(0) if hasattr(batch, 'y') and batch.y is not None else len(batch)
                total_loss += total_loss_value.item() * batch_size
                total_samples += batch_size
        
        avg_test_loss = total_loss / total_samples if total_samples > 0 else 0.0
        
        # 计算测试指标
        if len(all_predictions) > 0 and len(all_targets) > 0:
            test_metrics = calculate_metrics(
                np.array(all_targets), 
                np.array(all_predictions),
                threshold=self.eval_threshold  # 使用配置中的阈值
            )
            
            logger.info("=" * 50)
            logger.info("最终测试集评估结果")
            logger.info("=" * 50)
            logger.info(f"测试损失: {avg_test_loss:.4f}")
            logger.info(f"测试样本数: {total_samples}")
            logger.info(f"测试指标: {self._format_metrics(test_metrics)}")
            logger.info("=" * 50)
            
            # 保存测试结果
            test_results = {
                'test_loss': avg_test_loss,
                'test_samples': total_samples,
                'test_metrics': test_metrics,
                'model_path': str(best_model_path),
                'stage_metrics': self.best_metrics
            }
            
            results_path = self.checkpoint_dir / "final_test_results.json"
            import json
            with open(results_path, 'w') as f:
                json.dump(test_results, f, indent=2)
            logger.info(f"测试结果已保存到: {results_path}")
        
        else:
            logger.error("测试集评估失败：无法收集预测结果")
    
    def _format_metrics(self, metrics):
        """格式化指标显示"""
        formatted = []
        key_order = ['accuracy', 'precision', 'recall', 'f1', 'mcc', 'auc', 'aupr']
        
        for key in key_order:
            # 尝试不同的键名格式
            value = None
            if key in metrics:
                value = metrics[key]
            elif f'val_{key}' in metrics:
                value = metrics[f'val_{key}']
            elif f'test_{key}' in metrics:
                value = metrics[f'test_{key}']
            
            if value is not None:
                formatted.append(f"{key}: {value:.4f}")
        
        return ", ".join(formatted)


def load_data_loaders(config):
    """加载数据加载器，支持分布式训练"""
    try:
        # 尝试导入数据集模块
        from training.datasets import create_data_loaders
        
        # 获取数据路径配置
        data_config = {
            'data_root': config.get('paths', {}).get('data_root', './data'),
            'batch_size': config.get('training', {}).get('batch_size', 32),
            'num_workers': config.get('training', {}).get('num_workers', 4),
            'pin_memory': config.get('training', {}).get('pin_memory', True),
            'model': config.get('model', {})  # 添加模型配置，用于消融实验检测
        }
        
        # 创建数据加载器，包括测试集
        loaders = create_data_loaders(data_config)
        
        # 根据返回值数量判断是否包含测试集
        if len(loaders) == 3:
            train_loader, val_loader, test_loader = loaders
        else:
            train_loader, val_loader = loaders
            test_loader = None
        
        return train_loader, val_loader, test_loader
        
    except ImportError as e:
        logger.error(f"数据集模块导入失败: {e}")
        logger.error("请确保数据集模块正确安装和配置")
        raise e
    except Exception as e:
        logger.error(f"数据加载失败: {e}")
        raise e


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='SUCF-AMP训练脚本')
    parser.add_argument('--config', type=str, required=True,
                       help='训练配置文件路径')
    parser.add_argument('--resume', type=str, default=None,
                       help='恢复训练的检查点路径')
    parser.add_argument('--test-mode', action='store_true',
                       help='测试模式：使用少量数据测试整个流程')
    
    args = parser.parse_args()
    
    try:
        # 所有进程都报告初始状态
        logger.info("主进程开始初始化训练器")
        
        # 创建训练器
        trainer = SUCFTrainer(args.config)
        
        logger.info("训练器创建完成，开始加载数据")
        
        # 加载数据
        train_loader, val_loader, test_loader = load_data_loaders(trainer.config)
        
        logger.info("数据加载完成")
        
        if train_loader is None or val_loader is None:
            logger.error("数据加载失败，请检查数据加载逻辑")
            return
        
        logger.info("数据验证通过，准备开始训练")
        
        # 测试模式：限制数据量
        if args.test_mode:
            logger.info("运行测试模式：使用少量数据进行流程验证")
            # 将epoch数减少到2进行快速测试
            original_config = trainer.config['training']['sub_stages']
            for stage_name in original_config:
                original_config[stage_name]['epochs'] = min(2, original_config[stage_name].get('epochs', 2))
            logger.info("测试模式：已将所有阶段的epochs设置为2")
        
        logger.info("准备开始训练")
        
        # 开始训练
        trainer.train(train_loader, val_loader, test_loader)
        
        logger.info("训练成功完成！")
        
    except Exception as e:
        logger.error(f"进程 训练过程中发生错误: {str(e)}")
        import traceback
        logger.error(f"进程 错误堆栈: {traceback.format_exc()}")
        raise
    finally:
        pass


if __name__ == "__main__":
    main()
