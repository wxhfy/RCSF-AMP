#!/usr/bin/env python3
"""
SUCF Training Script - Simplified Version
Supports two-stage training: modal alignment pre-warming + collaborative fusion and prediction.
"""

import json
import os
import sys
import argparse
import logging
import time
import random
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
import shutil
from torch_geometric.loader import DataLoader as PyGDataLoader
from tqdm.auto import tqdm

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from models.sucf_model import create_sucf_model
from utils.sucf_losses import create_sucf_loss_function
from utils.metrics import calculate_metrics
from utils.early_stopping import EarlyStopping
from utils.config_utils import load_config, validate_config
from utils.datasets import create_data_loaders
# Setup logger
logger = logging.getLogger(__name__)


def set_random_seed(seed):
    """Sets random seed for all libraries to ensure reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Additional settings for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Set hash seed for Python
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    logger.info(f"Random seed set to: {seed}")


class SUCFTrainer:
    """SUCF Trainer, supporting two-stage training."""

    def __init__(self, config_path):
        """Initializes the trainer."""
        self.config_path = Path(config_path).resolve()
        self.config = load_config(self.config_path)
        validate_config(self.config)

        # Set random seed for reproducibility
        seed = self.config.get('random_seed', 42)
        set_random_seed(seed)

        self.run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.device = self._select_device()
        
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
        self.stage_thresholds = {}
        
        logger.info("SUCF Trainer initialized successfully.")

    def _progress_bar(self, iterable, description):
        """Creates a tqdm progress bar with consistent console behavior."""
        return tqdm(
            iterable,
            desc=description,
            dynamic_ncols=True,
            mininterval=0.3,
            file=sys.stdout,
            leave=False
        )

    def _create_output_dirs(self):
        """Creates output directories for checkpoints and logs from the config file."""
        paths = self.config.get('paths', {})
        self.checkpoint_dir = Path(paths.get('checkpoint_dir', './outputs/checkpoints'))
        self.log_dir = Path(paths.get('log_dir', './outputs/logs'))
        
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def _setup_logging(self):
        """Sets up logging to write to a file in the log directory."""
        log_file_path = self.log_dir / f"training_{self.run_id}.log"
        file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        root_logger = logging.getLogger()
        root_logger.handlers.clear()
        root_logger.addHandler(file_handler)
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
        root_logger.setLevel(logging.INFO)
        self.log_file_path = log_file_path
        logger.info(f"Logging to file: {log_file_path}")

    def _select_device(self):
        """Selects the compute device, honoring any config override."""
        training_cfg = self.config.get('training', {})
        device_override = training_cfg.get('device')

        if device_override:
            device = torch.device(device_override)
            if device.type == 'cuda' and not torch.cuda.is_available():
                raise RuntimeError(f"CUDA override requested ({device_override}) but CUDA is unavailable.")
        else:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if device.type == 'cuda':
            torch.cuda.set_device(device)

        return device
    
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

    def _search_best_threshold(self, targets, probabilities, metric_name='mcc'):
        """Finds the threshold that maximizes the specified metric on validation data."""
        if targets.size == 0:
            return 0.5, {}

        probabilities = probabilities.astype(float)
        targets = targets.astype(int)

        unique_scores = np.unique(probabilities)
        if unique_scores.size == 1:
            candidate_thresholds = np.array([0.5])
        else:
            mid_points = (unique_scores[:-1] + unique_scores[1:]) / 2.0
            candidate_thresholds = np.concatenate(([0.0], mid_points, [0.999999]))

        best_threshold = 0.5
        best_metric = -float('inf')
        best_metrics_snapshot = None

        for threshold in candidate_thresholds:
            metrics = calculate_metrics(targets, probabilities, threshold=threshold)
            metric_value = metrics.get(metric_name, 0.0)
            if metric_value > best_metric:
                best_metric = metric_value
                best_threshold = float(threshold)
                best_metrics_snapshot = metrics

        return best_threshold, best_metrics_snapshot or {}
    
    def train_epoch(self, train_loader, stage_config):
        """Trains the model for one epoch."""
        self.model.train()
        total_loss, total_samples = 0.0, 0
        
        stage_info = {
            'active_losses': stage_config.get('active_losses', ['activity']),
            'loss_weights': stage_config.get('loss_weights', {'activity': 1.0})
        }
        
        with self._progress_bar(train_loader, f'Stage {self.current_stage} Train') as pbar:
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
        """Validates the model for one epoch and collects logits for threshold search."""
        self.model.eval()
        total_loss, total_samples = 0.0, 0
        all_probs, all_targets = [], []
        
        stage_info = {
            'active_losses': stage_config.get('active_losses', ['activity']),
            'loss_weights': stage_config.get('loss_weights', {'activity': 1.0})
        }
        
        with torch.no_grad():
            with self._progress_bar(val_loader, f'Stage {self.current_stage} Validate') as pbar:
                for batch in pbar:
                    batch = batch.to(self.device)
                    outputs = self.model(batch)
                    
                    losses = self.loss_fn(outputs, batch, stage_info)
                    total_loss += losses['total_loss'].item() * batch.num_graphs
                    total_samples += batch.num_graphs
                    
                    probs = torch.sigmoid(outputs['activity_pred']).detach().cpu().numpy()
                    targets = batch.y.detach().cpu().numpy()
                    all_probs.append(probs)
                    all_targets.append(targets)
        
        val_metrics = {'val_loss': total_loss / total_samples if total_samples > 0 else 0.0}
        aux_data = {
            'probabilities': np.concatenate(all_probs) if all_probs else np.array([]),
            'targets': np.concatenate(all_targets).astype(int) if all_targets else np.array([], dtype=int)
        }

        if aux_data['probabilities'].size > 0:
            metrics = calculate_metrics(aux_data['targets'], aux_data['probabilities'])
            val_metrics.update({f'val_{k}': v for k, v in metrics.items()})

            best_thr, thr_metrics = self._search_best_threshold(aux_data['targets'], aux_data['probabilities'])
            val_metrics['val_best_threshold'] = best_thr
            val_metrics['val_best_threshold_mcc'] = thr_metrics.get('mcc', 0.0) if thr_metrics else 0.0
            aux_data['best_threshold'] = best_thr
            aux_data['best_threshold_metrics'] = thr_metrics
        else:
            aux_data['best_threshold'] = 0.5
            aux_data['best_threshold_metrics'] = {}
            val_metrics['val_best_threshold'] = 0.5
            val_metrics['val_best_threshold_mcc'] = 0.0
        
        return val_metrics, aux_data

    def save_checkpoint(self, checkpoint_name, metrics=None, is_best=False):
        """Saves a training checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'current_stage': self.current_stage,
            'current_epoch': self.current_epoch,
            'random_seed': self.config.get('random_seed', 42),
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
        # Restore random seed if available
        if 'random_seed' in checkpoint:
            set_random_seed(checkpoint['random_seed'])
            logger.info(f"Random seed restored from checkpoint: {checkpoint['random_seed']}")
        
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
        stage_best_threshold = None
        stage_best_threshold_metrics = None
        last_val_aux = None

        for epoch in range(stage_config.get('epochs', 10)):
            self.current_epoch = epoch + 1
            
            train_metrics = self.train_epoch(train_loader, stage_config)
            val_metrics, val_aux = self.validate_epoch(val_loader, stage_config)
            last_val_aux = val_aux
            
            log_message = (f"Stage {stage_name}, Epoch {self.current_epoch}: "
                           f"Train Loss: {train_metrics['train_loss']:.4f}, "
                           f"Val Loss: {val_metrics['val_loss']:.4f}, "
                           f"{self._format_metrics(val_metrics)}")

            if 'val_best_threshold' in val_metrics:
                log_message += f", Val Best Thr: {val_metrics['val_best_threshold']:.4f}"
            logger.info(log_message)
            
            if early_stopping:
                monitor_metric = val_metrics.get(monitor_key, float('inf'))
                
                should_stop = early_stopping(monitor_metric, epoch=self.current_epoch)
                
               
                if monitor_metric == early_stopping.get_best_score():
                    self.save_checkpoint(f"best_model_stage_{stage_name}", val_metrics, is_best=True)
                    stage_best_threshold = val_aux.get('best_threshold', stage_best_threshold)
                    stage_best_threshold_metrics = val_aux.get('best_threshold_metrics', stage_best_threshold_metrics)

                if should_stop:
                    break

        logger.info(f"--- Stage {stage_name} Finished ---")

        if stage_best_threshold is None and last_val_aux is not None:
            stage_best_threshold = last_val_aux.get('best_threshold', 0.5)
            stage_best_threshold_metrics = last_val_aux.get('best_threshold_metrics', {})

        self.stage_thresholds[stage_name] = {
            'threshold': stage_best_threshold if stage_best_threshold is not None else 0.5,
            'metrics': stage_best_threshold_metrics or {}
        }
        logger.info(
            f"Stage {stage_name} best dynamic threshold: {self.stage_thresholds[stage_name]['threshold']:.4f}"
        )

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
        
        test_metrics, test_aux = self.validate_epoch(
            test_loader, self.config['training']['sub_stages'][last_stage]
        )

        dynamic_threshold = self.stage_thresholds.get(last_stage, {}).get('threshold', 0.5)
        if test_aux['probabilities'].size > 0:
            dynamic_metrics = calculate_metrics(
                test_aux['targets'], test_aux['probabilities'], threshold=dynamic_threshold
            )
        else:
            dynamic_metrics = {}

        logger.info("=" * 50)
        logger.info("Final Test Set Results")
        logger.info("=" * 50)
        logger.info(f"Test Loss: {test_metrics.get('val_loss', 0.0):.4f}")
        logger.info(f"Default Threshold (0.5) Metrics: {self._format_metrics(test_metrics)}")
        logger.info(
            f"Dynamic Threshold ({dynamic_threshold:.4f}) Metrics: "
            f"{self._format_metrics({f'val_{k}': v for k, v in dynamic_metrics.items()})}"
        )
        logger.info("=" * 50)
        
        results_payload = {
            'config': str(self.config_path),
            'run_id': self.run_id,
            'default_threshold': 0.5,
            'default_metrics': test_metrics,
            'dynamic_threshold': dynamic_threshold,
            'dynamic_metrics': dynamic_metrics
        }
        
        results_filename = f"final_test_results_{self.config_path.stem}_{self.run_id}.json"
        results_path = self.log_dir / results_filename
        with open(results_path, 'w') as f:
            json.dump(results_payload, f, indent=2)
        logger.info(f"Test results saved to: {results_path}")

        latest_path = self.log_dir / "final_test_results_latest.json"
        try:
            shutil.copyfile(results_path, latest_path)
        except OSError:
            pass
    
    def _format_metrics(self, metrics):
        """Formats metrics for logging."""
        key_order = ['accuracy', 'precision', 'recall', 'f1', 'mcc', 'auc', 'aupr']
        return ", ".join([f"{key}: {metrics.get(f'val_{key}', 0):.4f}" for key in key_order if f'val_{key}' in metrics])


def load_data_loaders(config):
    """Loads training, validation, and test data loaders."""
    try:
        # Ensure reproducible data loading
        seed = config.get('random_seed', 42)
        
        data_config = {
            'data_root': config.get('paths', {}).get('data_root', './data'),
            'batch_size': config.get('training', {}).get('batch_size', 32),
            'num_workers': config.get('training', {}).get('num_workers', 4),
            'pin_memory': config.get('training', {}).get('pin_memory', True),
            'random_seed': seed,  # Pass seed to data loader creation
            'data': config.get('data', {})
        }
        return create_data_loaders(data_config)
    except Exception as e:
        logger.error(f"Failed to load data: {e}", exc_info=True)
        raise

def main():
    """Main function to run the training script."""
    parser = argparse.ArgumentParser(description='SUCF Training Script')
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