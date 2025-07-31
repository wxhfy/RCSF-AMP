#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Evaluation Metrics Utility
Contains functions for calculating all metrics used in model evaluation.
"""

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, matthews_corrcoef,
    confusion_matrix
)
from typing import Dict, Union
import logging

logger = logging.getLogger(__name__)


def calculate_metrics(y_true: Union[np.ndarray, torch.Tensor],
                      y_scores: Union[np.ndarray, torch.Tensor],
                      threshold: float = 0.5) -> Dict[str, float]:
    """
    Calculates a comprehensive set of metrics for a binary classification task.
    
    Args:
        y_true (np.ndarray or torch.Tensor): The ground truth labels (integers 0 or 1).
        y_scores (np.ndarray or torch.Tensor): The predicted probabilities or scores (floats between 0 and 1).
        threshold (float): The threshold for converting probabilities to binary predictions.
        
    Returns:
        Dict[str, float]: A dictionary containing all calculated metrics.
    """
    # --- 1. Input Validation and Conversion ---
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_scores, torch.Tensor):
        y_scores = y_scores.detach().cpu().numpy()

    y_true = y_true.astype(int)
    # Convert scores to binary predictions based on the threshold
    y_pred = (y_scores > threshold).astype(int)

    metrics = {}

    # --- 2. Calculate Metrics ---
    try:
        # Standard classification metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
        metrics['f1'] = f1_score(y_true, y_pred, zero_division=0)
        
        # Matthews Correlation Coefficient
        try:
            metrics['mcc'] = matthews_corrcoef(y_true, y_pred)
        except ValueError:
            # MCC is undefined for some inputs (e.g., all true/false positives/negatives)
            metrics['mcc'] = 0.0

        # Confusion matrix components
        try:
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            metrics['tn'], metrics['fp'], metrics['fn'], metrics['tp'] = int(tn), int(fp), int(fn), int(tp)
            # Specificity (True Negative Rate)
            metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            # Sensitivity is the same as recall
            metrics['sensitivity'] = metrics['recall']
        except ValueError:
            # Handle cases where confusion matrix is not 2x2
            metrics.update({'tn': 0, 'fp': 0, 'fn': 0, 'tp': 0, 'specificity': 0.0, 'sensitivity': 0.0})

        # Score-based metrics (AUC, AUPR)
        try:
            # Check if there is more than one class in the true labels
            if len(np.unique(y_true)) > 1:
                metrics['auc'] = roc_auc_score(y_true, y_scores)
                metrics['aupr'] = average_precision_score(y_true, y_scores)
            else:
                # AUC/AUPR are not defined for a single class
                metrics['auc'] = 0.5
                metrics['aupr'] = 0.5
        except ValueError as e:
            logger.warning(f"Could not calculate AUC/AUPR metrics: {e}. Defaulting to 0.5.")
            metrics['auc'] = 0.5
            metrics['aupr'] = 0.5

        return {k: float(v) for k, v in metrics.items()}

    except Exception as e:
        logger.error(f"An unexpected error occurred during metric calculation: {e}", exc_info=True)
        # Return a dictionary of zeros in case of a major failure
        return {
            'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'sensitivity': 0.0,
            'specificity': 0.0, 'f1': 0.0, 'mcc': 0.0, 'auc': 0.0, 'aupr': 0.0,
            'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0
        }