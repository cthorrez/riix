"""utils for evaluating rating systems"""

import time
from copy import deepcopy
from multiprocessing import Pool
import numpy as np
from riix.core.base import OnlineRatingSystem
from riix.utils.data_utils import TimedPairDataset
from riix.metrics import binary_metrics_suite


def evaluate(model: OnlineRatingSystem, dataset: TimedPairDataset, metrics_mask: np.ndarray = None):
    """evaluate a rating system on a dataset"""
    start_time = time.time()
    if metrics_mask is None:
        metrics_mask = np.ones(len(dataset), dtype=np.bool_)
    probs = model.fit_dataset(dataset, return_pre_match_probs=True)[metrics_mask]
    outcomes = dataset.outcomes[metrics_mask]
    duration = time.time() - start_time
    metrics = binary_metrics_suite(probs, outcomes)
    metrics['duration'] = duration
    return metrics


def eval_wrapper(params, rating_system_class, dataset, metrics_mask):
    model = rating_system_class(competitors=dataset.competitors, **params)
    return evaluate(model, dataset, metrics_mask)


# grid_search worker state, set once per worker process by _init_grid_search_worker so that only
# the (small) param dicts are pickled per task rather than the dataset itself
_worker_state = {}


def _init_grid_search_worker(rating_system_class, dataset, metrics_mask):
    _worker_state['rating_system_class'] = rating_system_class
    _worker_state['dataset'] = dataset
    _worker_state['metrics_mask'] = metrics_mask


def _grid_search_task(params):
    return eval_wrapper(
        params, _worker_state['rating_system_class'], _worker_state['dataset'], _worker_state['metrics_mask']
    )


def grid_search(
    rating_system_class,
    dataset,
    param_configurations,
    metrics_mask=None,
    metric='log_loss',
    minimize_metric=True,
    num_processes=None,
    return_all_metrics=False,
):
    """Perform grid search and return the best hyperparameters."""
    best_params = {}
    best_metrics = {}
    metric_multiplier = 1.0 if minimize_metric else -1.0
    best_metric = np.inf

    if num_processes:
        with Pool(
            num_processes,
            initializer=_init_grid_search_worker,
            initargs=(rating_system_class, dataset, metrics_mask),
        ) as pool:
            all_metrics = pool.map(_grid_search_task, param_configurations)
    else:
        all_metrics = [
            eval_wrapper(params, rating_system_class, dataset, metrics_mask) for params in param_configurations
        ]

    for current_params, current_metrics in zip(param_configurations, all_metrics):
        current_metric = current_metrics[metric]
        # Compare and update best metric and params
        if current_metric * metric_multiplier < best_metric:
            best_metric = current_metric * metric_multiplier
            best_metrics = deepcopy(current_metrics)
            best_params = current_params

    if not return_all_metrics:
        return best_params, best_metrics
    return best_params, best_metrics, all_metrics
