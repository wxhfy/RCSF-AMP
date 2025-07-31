#!/usr/bin/env python3
"""
SUCF-AMP Training Script - Simplified Version
Supports two-stage training: modal alignment pre-warming + collaborative fusion and prediction.
"""

import json
import os
import sys
import argparse
import logging
import time
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
from torch_geometric.loader import DataLoader as PyGDataLoader
from tqdm import tqdm

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from models.sucf_model import create_sucf_model
from training.sucf_losses import create_sucf_loss_function
from utils.metrics import calculate_metrics
from utils.early_stopping import EarlyStopping
from utils.config_utils import load_config, validate_config, get_evaluation_threshold
from utils.datasets import create_data_loaders
# Setup logger
logger = logging.getLogger(__name__)


class SUCFTrainer:
    """SUCF Trainer, supporting two-stage training."""

    def __init__(self, config_path):
        """Initializes the trainer."""
        self.config = load_config(config_path)
        validate_config(self.config)

        self.eval_threshold = get_evaluation_threshold(self.config)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self._create_output_dirs()
        self._setup_logging()
        
        logger.info(f"Using device: {self.device}")
        
        self.model = self._create_model()
        self.loss_fn = create_sucf_loss_function(self.config)
        self.optimizer = None
        self.scheduler = None
        self.current_stage = None
        self.current_epoch = 0
        self.best_metrics = {}
        
        logger.info("SUCF Trainer initialized successfully.")

    def _create_output_dirs(self):
        """Creates output directories for checkpoints and logs from the config file."""
        paths = self.config.get('paths', {})
        self.checkpoint_dir = Path(paths.get('checkpoint_dir', './outputs/checkpoints'))
        self.log_dir = Path(paths.get('log_dir', './outputs/logs'))
        
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def _setup_logging(self):
        """Sets up logging to write to a file in the log directory."""
        log_file_path = self.log_dir / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        root_logger = logging.getLogger()
        root_logger.handlers.clear()
        root_logger.addHandler(file_handler)
        root_logger.addHandler(logging.StreamHandler(sys.stdout))
        root_logger.setLevel(logging.INFO)
        logger.info(f"Logging to file: {log_file_path}")
    
    def _create_model(self):
        """Creates the SUCF model."""
        model_config = self.config.get('model', {})
        model = create_sucf_model(model_config).to(self.device)
        
        model_info = model.get_model_info()
        logger.info(f"Total model parameters: {model_info['total_params']:,}")
        logger.info(f"Trainable parameters: {model_info['trainable_params']:,}")
        
        return model
    
    def _create_optimizer_and_scheduler(self, stage_config):
        """Creates optimizer and scheduler for the current training stage."""
        optimizer_stage_config = stage_config.get('optimizer', {})
        default_lr = float(optimizer_stage_config.get('lr', 1e-3))
        head_lr = float(optimizer_stage_config.get('head_lr', default_lr))
        
        head_params = [p for n, p in self.model.named_parameters() if 'activity_predictor' in n or 'global_pooling' in n]
        head_param_ids = {id(p) for p in head_params}
        other_params = [p for p in self.model.parameters() if id(p) not in head_param_ids]
        
        param_groups = [
            {'params': head_params, 'lr': head_lr, 'name': 'head_params'},
            {'params': other_params, 'lr': default_lr, 'name': 'default'}
        ]
        
        self.optimizer = torch.optim.AdamW(param_groups, weight_decay=self.config['training']['optimizer_params']['weight_decay'])
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=stage_config.get('epochs', 100), eta_min=default_lr * self.config['training']['scheduler_params']['min_lr_factor']
        )
        logger.info(f"Created Optimizer: AdamW, Default LR: {default_lr}, Head LR: {head_lr}")
    
    def train_epoch(self, train_loader, stage_config):
        """Trains the model for one epoch."""
        self.model.train()
        total_loss, total_samples = 0.0, 0
        
        stage_info = {
            'active_losses': stage_config.get('active_losses', ['activity']),
            'loss_weights': stage_config.get('loss_weights', {'activity': 1.0})
        }
        
        pbar = tqdm(train_loader, desc=f'Stage {self.current_stage} Train', ncols=100)
        for batch in pbar:
            batch = batch.to(self.device)
            
            outputs = self.model(batch)
            losses = self.loss_fn(outputs, batch, stage_info)
            total_loss_value = losses['total_loss']
            
            self.optimizer.zero_grad()
            total_loss_value.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            batch_size = batch.num_graphs
            total_loss += total_loss_value.item() * batch_size
            total_samples += batch_size
            
            pbar.set_postfix({'loss': f"{total_loss/total_samples:.4f}"})
        
        if self.scheduler:
            self.scheduler.step()
            
        return {'train_loss': total_loss / total_samples if total_samples > 0 else 0.0}
    
    def validate_epoch(self, val_loader, stage_config):
        """Validates the model for one epoch."""
        self.model.eval()
        total_loss, total_samples = 0.0, 0
        all_preds, all_targets = [], []
        
        stage_info = {
            'active_losses': stage_config.get('active_losses', ['activity']),
            'loss_weights': stage_config.get('loss_weights', {'activity': 1.0})
        }
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f'Stage {self.current_stage} Validate', ncols=100):
                batch = batch.to(self.device)
                outputs = self.model(batch)
                
                losses = self.loss_fn(outputs, batch, stage_info)
                total_loss += losses['total_loss'].item() * batch.num_graphs
                total_samples += batch.num_graphs
                
                all_preds.extend(torch.sigmoid(outputs['activity_pred']).cpu().numpy())
                all_targets.extend(batch.y.cpu().numpy())
        
        val_metrics = {'val_loss': total_loss / total_samples if total_samples > 0 else 0.0}
        if all_preds:
            metrics = calculate_metrics(np.array(all_targets), np.array(all_preds), threshold=self.eval_threshold)
            val_metrics.update({f'val_{k}': v for k, v in metrics.items()})
        
        return val_metrics

    def save_checkpoint(self, checkpoint_name, metrics=None, is_best=False):
        """Saves a training checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'current_stage': self.current_stage,
            'current_epoch': self.current_epoch,
            'eval_threshold': self.eval_threshold,
            'metrics': metrics or {},
            'config': self.config
        }
        
        checkpoint_path = self.checkpoint_dir / f"{checkpoint_name}.pth"
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_path}")
        
        if is_best:
            best_path = self.checkpoint_dir / f"best_model_stage_{self.current_stage}.pth"
            torch.save(checkpoint, best_path)
            logger.info(f"Best model for stage {self.current_stage} saved: {best_path}")
    
    def load_checkpoint(self, checkpoint_path):
        """Loads a training checkpoint, without optimizer/scheduler states."""
        if not os.path.exists(checkpoint_path):
            logger.warning(f"Checkpoint file not found: {checkpoint_path}. Skipping load.")
            return
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.eval_threshold = checkpoint.get('eval_threshold', self.eval_threshold)
        logger.info(f"Model state loaded from {checkpoint_path}.")
    
    def train_stage(self, stage_name, stage_config, train_loader, val_loader):
        """Trains a single, complete stage."""
        self.current_stage = stage_name
        logger.info(f"--- Starting Training Stage: {stage_name} ({stage_config.get('description', 'N/A')}) ---")
        
        self._create_optimizer_and_scheduler(stage_config)
        
        early_stopping_config = stage_config.get('early_stopping', {})
        early_stopping = EarlyStopping(
            patience=early_stopping_config.get('patience', 10),
            mode=early_stopping_config.get('mode', 'min')
        ) if early_stopping_config.get('enable', False) else None
        
        monitor_key = early_stopping_config.get('monitor', 'val_loss')

        for epoch in range(stage_config.get('epochs', 10)):
            self.current_epoch = epoch + 1
            
            train_metrics = self.train_epoch(train_loader, stage_config)
            val_metrics = self.validate_epoch(val_loader, stage_config)
            
            log_message = (f"Stage {stage_name}, Epoch {self.current_epoch}: "
                           f"Train Loss: {train_metrics['train_loss']:.4f}, "
                           f"Val Loss: {val_metrics['val_loss']:.4f}, "
                           f"{self._format_metrics(val_metrics)}")
            logger.info(log_message)
            
            if early_stopping:
                monitor_metric = val_metrics.get(monitor_key, float('inf'))
                
                should_stop = early_stopping(monitor_metric, epoch=self.current_epoch)
                
               
                if monitor_metric == early_stopping.get_best_score():
                    self.save_checkpoint(f"best_model_stage_{stage_name}", val_metrics, is_best=True)

                if should_stop:
                    break

        logger.info(f"--- Stage {stage_name} Finished ---")

    def train(self, train_loader, val_loader, test_loader=None):
        """Executes the full two-stage training pipeline."""
        logger.info("Starting two-stage training pipeline.")
        
        training_stages = self.config.get('training', {}).get('sub_stages', {})
        
        for stage_name in sorted(training_stages.keys()):
            stage_config = training_stages[stage_name]
            
            if stage_name == "2.0":
                prev_stage_best_model = self.checkpoint_dir / "best_model_stage_1.0.pth"
                if prev_stage_best_model.exists():
                    logger.info(f"Loading best model from Stage 1.0: {prev_stage_best_model}")
                    self.load_checkpoint(str(prev_stage_best_model))
                else:
                    logger.warning("Could not find best model from Stage 1.0. Continuing with current weights.")
            
            self.train_stage(stage_name, stage_config, train_loader, val_loader)
        
        if test_loader:
            self.evaluate_test_set(test_loader)
        else:
            logger.info("Training complete. No test set provided for final evaluation.")

    def evaluate_test_set(self, test_loader):
        """Evaluates the final model on the test set."""
        logger.info("--- Starting Final Evaluation on Test Set ---")
        
        last_stage = max(self.config.get('training', {}).get('sub_stages', {}).keys())
        best_model_path = self.checkpoint_dir / f"best_model_stage_{last_stage}.pth"
        
        if best_model_path.exists():
            logger.info(f"Loading best model for testing: {best_model_path}")
            self.load_checkpoint(str(best_model_path))
        else:
            logger.warning("No best model checkpoint found. Using the current model state for testing.")
        
        test_metrics = self.validate_epoch(test_loader, self.config['training']['sub_stages'][last_stage])
        
        logger.info("=" * 50)
        logger.info("Final Test Set Results")
        logger.info("=" * 50)
        logger.info(f"Test Loss: {test_metrics.get('val_loss', 0):.4f}")
        logger.info(f"Test Metrics (threshold={self.eval_threshold}): {self._format_metrics(test_metrics)}")
        logger.info("=" * 50)
        
        results_path = self.log_dir / "final_test_results.json"
        with open(results_path, 'w') as f:
            json.dump(test_metrics, f, indent=2)
        logger.info(f"Test results saved to: {results_path}")
    
    def _format_metrics(self, metrics):
        """Formats metrics for logging."""
        key_order = ['accuracy', 'precision', 'recall', 'f1', 'mcc', 'auc', 'aupr']
        return ", ".join([f"{key}: {metrics.get(f'val_{key}', 0):.4f}" for key in key_order if f'val_{key}' in metrics])


def load_data_loaders(config):
    """Loads training, validation, and test data loaders."""
    try:
        data_config = {
            'data_root': config.get('paths', {}).get('data_root', './data'),
            'batch_size': config.get('training', {}).get('batch_size', 32),
            'num_workers': config.get('training', {}).get('num_workers', 4)
        }
        return create_data_loaders(data_config)
    except Exception as e:
        logger.error(f"Failed to load data: {e}", exc_info=True)
        raise

def main():
    """Main function to run the training script."""
    parser = argparse.ArgumentParser(description='SUCF-AMP Training Script')
    parser.add_argument('--config', type=str, required=True, help='Path to the training configuration file.')
    parser.add_argument('--test-mode', action='store_true', help='Run in test mode with minimal epochs for validation.')
    
    args = parser.parse_args()
    
    try:
        trainer = SUCFTrainer(args.config)

        # In test mode, modify the config to run a quick test
        if args.test_mode:
            logger.info("--- RUNNING IN TEST MODE ---")
            for stage_name in trainer.config['training']['sub_stages']:
                trainer.config['training']['sub_stages'][stage_name]['epochs'] = 2
            logger.info("All stage epochs have been set to 2 for a quick run.")

        train_loader, val_loader, test_loader = load_data_loaders(trainer.config)
        
        trainer.train(train_loader, val_loader, test_loader)
        logger.info("Training finished successfully!")
        
    except Exception as e:
        logger.error(f"An error occurred during training: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()