"""utils for evaluating rating systems"""
import time
from itertools import product
from collections import defaultdict
import numpy as np
from riix.core.base import OnlineRatingSystem
from riix.utils.data_utils import RatingDataset
from riix.metrics import binary_metrics_suite


def evaluate(rater: OnlineRatingSystem, dataset: RatingDataset, cache=True):
    """evaluate a rating system on a dataset"""
    all_probs = np.zeros(len(dataset))
    probs_idx = 0
    start_time = time.time()
    for time_step, matchups, outcomes in dataset:
        probs = rater.predict(time_step=time_step, matchups=matchups, set_cache=cache)
        all_probs[probs_idx : probs_idx + probs.shape[0]] = probs
        probs_idx += probs.shape[0]
        rater.fit(time_step, matchups, outcomes, use_cache=cache)
    duration = time.time() - start_time
    metrics = binary_metrics_suite(all_probs, dataset.outcomes)
    metrics['duration'] = duration
    return metrics


def grid_search(rating_system_class, dataset, params_grid, metric='log_loss', minimize_metric=True):
    """Perform grid search and return the best hyperparameters."""
    best_params = {}
    best_metric = np.inf if minimize_metric else -np.inf
    metric_multiplier = 1.0 if minimize_metric else -1.0

    for setting in product(*params_grid.values()):
        current_params = dict(zip(params_grid.keys(), setting))
        rating_system = rating_system_class(num_competitors=dataset.num_competitors, **current_params)
        current_metrics = evaluate(rating_system, dataset)
        current_metric = current_metrics[metric] * metric_multiplier

        # Compare and update best metric and params
        if current_metric * metric_multiplier < best_metric * metric_multiplier:
            best_metric = current_metric
            best_params = current_params

    # Adjust best_metric back to its original scale
    best_metric *= metric_multiplier
    return best_params, best_metric


def add_mean_metrics(data_dict):
    """get overall mean for each rating system and metric at the game level"""
    # Initialize a structure to store sum and counts for calculating means
    rating_system_sums = defaultdict(lambda: defaultdict(float))
    rating_system_counts = defaultdict(lambda: defaultdict(int))

    # Collect sums and counts for each metric in each rating system
    for rating_systems in data_dict.values():
        for rating_system, metrics in rating_systems.items():
            for metric, value in metrics.items():
                rating_system_sums[rating_system][metric] += value
                rating_system_counts[rating_system][metric] += 1

    # Calculate mean metrics for each rating system
    mean_metrics = {
        sys: {metric: total / rating_system_counts[sys][metric] for metric, total in metrics.items()}
        for sys, metrics in rating_system_sums.items()
    }

    # Add the 'mean' key at the top level of the data_dict
    data_dict['mean'] = mean_metrics

    return data_dict
