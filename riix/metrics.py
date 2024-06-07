"""module for computing metrics for rating system experiments"""

import numpy as np


def binary_accuracy(probs: np.ndarray, outcomes: np.ndarray) -> float:
    """compute accuracy where outcomes is binary ties count for half"""
    pos_mask = probs > 0.5
    neg_mask = probs < 0.5
    draw_mask = probs == 0.5
    correct = outcomes[pos_mask].sum() + (1.0 - outcomes[neg_mask]).sum() + 0.5 * draw_mask.sum()
    return correct / probs.shape[0]

def accuracy_without_draws(probs: np.ndarray, outcomes: np.ndarray) -> float:
    """compute binary accuracy after first filtering out rows where the label is a draw"""
    draw_mask = outcomes == 0.5
    probs = probs[~draw_mask]
    outcomes = outcomes[~draw_mask]
    return binary_accuracy(probs, outcomes)


def accuracy_with_draws(probs: np.ndarray, outcomes: np.ndarray, draw_margin=0.0) -> float:
    """computes accuracy while allowing for ties"""
    pos_pred_mask = probs > (0.5 + draw_margin)
    neg_pred_mask = probs < (0.5 - draw_margin)
    draw_pred_mask = np.abs(probs - draw_margin) <= draw_margin
    correct = outcomes[pos_pred_mask].sum()
    correct += (1.0 - outcomes)[neg_pred_mask].sum()
    correct += 2 * outcomes[draw_pred_mask].sum()  # lmao
    return correct / outcomes.shape[0]


def binary_log_loss(probs: np.ndarray, outcomes: np.ndarray, eps: float = 1e-6) -> float:
    """compute log loss where outcome is binary 1.0 or 0.0"""
    probs = np.clip(probs, eps, 1 - eps)
    loss_array = -(np.log(probs) * outcomes) - (np.log(1.0 - probs) * (1.0 - outcomes))
    return loss_array.mean()


def brier_score(probs: np.ndarray, outcomes: np.ndarray) -> float:
    """compute the brier score, which is equivalent to the MSE"""
    return np.square(probs - outcomes).mean()


def binary_metrics_suite(probs: np.ndarray, outcomes: np.ndarray):
    """a wrapper class for running a bunch of binary metrics"""
    metrics = {
        'accuracy': binary_accuracy(probs, outcomes),
        'accuracy_without_draws' : accuracy_without_draws(probs, outcomes),
        'log_loss': binary_log_loss(probs, outcomes),
        'brier_score': brier_score(probs, outcomes),
    }
    return metrics
