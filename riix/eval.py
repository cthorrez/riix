"""utils for evaluating rating systems"""
import time
from functools import partial
from copy import deepcopy
from itertools import product
from multiprocessing import Pool
import numpy as np
from riix.core.base import OnlineRatingSystem
from riix.utils.data_utils import MatchupDataset
from riix.metrics import binary_metrics_suite


def evaluate(model: OnlineRatingSystem, dataset: MatchupDataset, test_dataset=None):
    """evaluate a rating system on a dataset"""
    start_time = time.time()
    if test_dataset is None:
        probs = model.fit_dataset(dataset, return_pre_match_probs=True)
        outcomes = dataset.outcomes
    if test_dataset is not None:
        model.fit_dataset(dataset, return_pre_match_probs=False)
        probs = model.fit_dataset(test_dataset, return_pre_match_probs=True)
        outcomes = test_dataset.outcomes
    duration = time.time() - start_time
    metrics = binary_metrics_suite(probs, outcomes)
    metrics['duration'] = duration
    return metrics


def eval_wrapper(params, rating_system_class, dataset, test_dataset=None):
    model = rating_system_class(competitors=dataset.competitors, **params)
    return evaluate(model, dataset, test_dataset=test_dataset)


def create_uniform_grid(
    param_ranges,
    num_samples,
    seed=0,
):
    rng = np.random.default_rng(seed)
    num_hyperparams = len(param_ranges)
    low = np.empty(num_hyperparams)
    high = np.empty(num_hyperparams)
    for idx, param_range in enumerate(param_ranges.values()):
        low[idx] = param_range[0]
        high[idx] = param_range[1]
    values = rng.uniform(low=low, high=high, size=(num_samples, num_hyperparams))
    params = []
    for sample_idx, values_row in enumerate(values):
        current_params = {}
        for param_idx, param_name in enumerate(param_ranges.keys()):
            current_params[param_name] = values_row[param_idx]
        params.append(current_params)
    return params


def grid_search(
    rating_system_class,
    dataset,
    test_dataset=None,
    param_grid=None,
    param_ranges=None,
    num_samples=None,
    metric='log_loss',
    minimize_metric=True,
    num_processes=None,
    seed=0,
    return_all_metrics=False,
):
    """Perform grid search and return the best hyperparameters."""
    map_fn = map
    if num_processes:
        pool = Pool(num_processes)
        map_fn = pool.map

    best_params = {}
    best_metrics = {}
    metric_multiplier = 1.0 if minimize_metric else -1.0
    best_metric = np.inf

    if (param_grid is not None) and (param_ranges is None):
        all_params = product(*param_grid.values())
        inputs = [dict(zip(param_grid.keys(), params)) for params in all_params]
    elif (param_ranges is not None) and (param_grid is None):
        inputs = create_uniform_grid(param_ranges, num_samples, seed)
        pass
    else:
        raise ValueError('you must only specify one of param_grid and param_ranges')

    func = partial(
        eval_wrapper,
        rating_system_class=rating_system_class,
        dataset=dataset,
    )
    all_metrics = list(map_fn(func, inputs))
    if num_processes:
        pool.close()
        pool.join()

    for current_params, current_metrics in zip(inputs, all_metrics):
        current_metric = current_metrics[metric]
        # Compare and update best metric and params
        if current_metric * metric_multiplier < best_metric:
            best_metric = current_metric * metric_multiplier
            best_metrics = deepcopy(current_metrics)
            best_params = current_params

    if not return_all_metrics:
        return best_params, best_metrics
    return best_params, best_metrics, all_metrics
