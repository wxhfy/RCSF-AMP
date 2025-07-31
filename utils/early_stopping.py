#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Early Stopping Utility
Used to monitor metric changes during training to prevent overfitting.
"""

import numpy as np
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class EarlyStopping:
    """
    Early stopping utility to monitor a validation metric and stop training when it ceases to improve.
    """
    
    def __init__(self, 
                 patience: int = 10,
                 min_delta: float = 0.0,
                 mode: str = 'min',
                 verbose: bool = True):
        """
        Initializes the EarlyStopping object.
        
        Args:
            patience (int): Number of epochs with no improvement after which training will be stopped.
            min_delta (float): Minimum change in the monitored quantity to qualify as an improvement.
            mode (str): One of {'min', 'max'}. In 'min' mode, training stops when the quantity monitored has stopped decreasing; in 'max' mode it stops when the quantity has stopped increasing.
            verbose (bool): If True, prints a message for each improvement.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose
        
        self.wait = 0
        self.stopped_epoch = 0
        self.best_score = None
        self.early_stop = False
        
        if mode == 'min':
            self.monitor_op = np.less
            self.min_delta *= -1
        elif mode == 'max':
            self.monitor_op = np.greater
        else:
            raise ValueError(f"Mode must be 'min' or 'max', but got: {mode}")

    def is_best_score(self, current_score: float) -> bool:
        """
        Checks if the current score is the best one seen so far.
        This method is now explicitly defined to fix the AttributeError.
        
        Args:
            current_score (float): The metric score of the current epoch.

        Returns:
            bool: True if the current score is the best, False otherwise.
        """
        if self.best_score is None:
            return True
        return self.monitor_op(current_score - self.min_delta, self.best_score)
    
    def __call__(self, current_score: float, epoch: Optional[int] = None) -> bool:
        """
        Checks if training should be stopped based on the current score.
        
        Args:
            current_score (float): The metric score of the current epoch.
            epoch (int, optional): The current epoch number for logging purposes.
            
        Returns:
            bool: True if training should stop, False otherwise.
        """
        if self.is_best_score(current_score):
            self.best_score = current_score
            self.wait = 0
            if self.verbose:
                logger.info(f"Epoch {epoch}: Monitored metric improved to {current_score:.4f}")
        else:
            self.wait += 1
            if self.verbose:
                logger.info(f"Epoch {epoch}: Metric did not improve (current: {current_score:.4f}, best: {self.best_score:.4f}). Patience: {self.wait}/{self.patience}")
            
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.early_stop = True
                if self.verbose:
                    logger.info(f"Early stopping triggered! Stopping at epoch {epoch}.")
                    logger.info(f"Best score achieved: {self.best_score:.4f}")
        
        return self.early_stop
    
    def reset(self):
        """Resets the state of the early stopper."""
        self.wait = 0
        self.stopped_epoch = 0
        self.best_score = None
        self.early_stop = False
        if self.verbose:
            logger.info("Early stopping state has been reset.")
    
    def get_best_score(self) -> Optional[float]:
        """Returns the best score found so far."""
        return self.best_score
    
    def should_stop(self) -> bool:
        """Returns whether the training should be stopped."""
        return self.early_stop