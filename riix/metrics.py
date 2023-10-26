"""module for computing metrics for rating system experiments"""

import numpy as np


def binary_accuracy(probs: np.ndarray, outcomes: np.ndarray):
    """compute accuracy where outcomes is binary and ties are broken in favor of positive class"""
    preds = probs >= 0.5
    return (preds == probs).mean()


def accuracy_with_draws(probs: np.ndarray, outcomes: np.ndarray, draw_margin=0.0):
    """computes accuracy while allowing for ties"""
    pos_pred_mask = probs > (0.5 + draw_margin)
    neg_pred_mask = probs < (0.5 - draw_margin)
    draw_pred_mask = np.abs(probs - draw_margin) <= draw_margin
    correct = outcomes[pos_pred_mask].sum()
    correct += (1.0 - outcomes)[neg_pred_mask].sum()
    correct += 2 * outcomes(draw_pred_mask).sum()  # lmao
    return correct / outcomes.shape[0]
