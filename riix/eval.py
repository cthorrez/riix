"""utils for evaluating rating systems"""
import time
from functools import partial
from copy import deepcopy
from itertools import product
from collections import defaultdict
from multiprocessing import Pool
import numpy as np
from riix.core.base import OnlineRatingSystem
from riix.utils.data_utils import MatchupDataset
from riix.metrics import binary_metrics_suite


def evaluate(model: OnlineRatingSystem, dataset: MatchupDataset, cache=True):
    """evaluate a rating system on a dataset"""
    start_time = time.time()
    probs = model.fit_dataset(dataset, return_pre_match_probs=True)
    duration = time.time() - start_time
    metrics = binary_metrics_suite(probs, dataset.outcomes)
    metrics['duration'] = duration
    return metrics


def train_and_evaluate(model: OnlineRatingSystem, train_dataset: MatchupDataset, test_dataset: MatchupDataset):
    """fit on train_dataset, evaluate on test_dataset"""
    start_time = time.time()
    model.fit_dataset(train_dataset)
    test_probs = model.fit_dataset(test_dataset, return_pre_match_probs=True)
    duration = time.time() - start_time
    metrics = binary_metrics_suite(test_probs, test_dataset.outcomes)
    metrics['duration'] = duration
    return metrics


def eval_wrapper(params, rating_system_class, train_dataset, test_dataset):
    model = rating_system_class(competitors=train_dataset.competitors, **params)
    return train_and_evaluate(model, train_dataset, test_dataset)


def grid_search(
    rating_system_class,
    train_dataset,
    test_dataset,
    params_grid,
    metric='log_loss',
    minimize_metric=True,
    num_processes=None,
):
    """Perform grid search and return the best hyperparameters."""
    map_fn = map
    if num_processes:
        pool = Pool(num_processes)
        map_fn = pool.imap

    best_params = {}
    best_metrics = {}
    metric_multiplier = 1.0 if minimize_metric else -1.0
    best_metric = np.inf

    all_params = product(*params_grid.values())
    inputs = [dict(zip(params_grid.keys(), params)) for params in all_params]

    func = partial(
        eval_wrapper,
        rating_system_class=rating_system_class,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
    )
    all_metrics = map_fn(func, inputs)

    for current_params, current_metrics in zip(inputs, all_metrics):
        current_metric = current_metrics[metric]
        # Compare and update best metric and params
        if current_metric * metric_multiplier < best_metric:
            best_metric = current_metric * metric_multiplier
            best_metrics = deepcopy(current_metrics)
            best_params = current_params

    return best_params, best_metrics


def multi_dataset_grid_search(rating_system_class, datasets, params_grid, metric='log_loss', minimize_metric=True):
    """
    Extends grid_search() to multiple datasets.
    Uses the mean of the desired metric across datasets for evaluation.
    """
    best_params = {}
    best_metrics = {}
    metric_multiplier = 1.0 if minimize_metric else -1.0
    best_metric = np.inf

    for setting in product(*params_grid.values()):
        current_params = dict(zip(params_grid.keys(), setting))
        metrics_across_datasets = []

        # Evaluate across all datasets
        for dataset in datasets:
            rating_system = rating_system_class(competitors=dataset.competitors, **current_params)
            current_metrics = evaluate(rating_system, dataset)
            metrics_across_datasets.append(current_metrics[metric])

        # Calculate the mean metric across datasets
        mean_metric = np.mean(metrics_across_datasets)
        print(current_params)
        print(mean_metric)

        # Compare and update the best metric and params
        if mean_metric * metric_multiplier < best_metric:
            best_metric = mean_metric * metric_multiplier
            best_metrics = {'mean_' + metric: mean_metric}
            best_params = current_params

    return best_params, best_metrics


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
