"""utils for evaluating rating systems and searching over their hyperparameters"""

import math
import time
from copy import deepcopy
from multiprocessing import get_context
import numpy as np
from riix.core.base import RatingSystem
from riix.utils.data_utils import TimedPairDataset
from riix.metrics import binary_metrics_suite


def evaluate(model: RatingSystem, dataset: TimedPairDataset, metrics_mask: np.ndarray = None):
    """evaluate a rating system on a dataset"""
    start_time = time.time()
    if metrics_mask is None:
        metrics_mask = np.ones(len(dataset), dtype=np.bool_)
    probs = model.fit_dataset(dataset, return_pre_match_probs=True).probs[metrics_mask]
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


def _run_evals(
    rating_system_class,
    dataset,
    param_configurations,
    metrics_mask,
    metric,
    metric_multiplier,
    num_processes,
    verbose,
):
    """Evaluate every configuration, optionally in parallel, with optional progress reporting."""
    total = len(param_configurations)
    all_metrics = []
    running_best = np.inf

    def report(idx, params, current_metrics):
        nonlocal running_best
        if not verbose:
            return
        current = current_metrics[metric] * metric_multiplier
        running_best = min(running_best, current)
        print(
            f'[{idx + 1}/{total}] {metric}={current_metrics[metric]:.6f} '
            f'best={running_best * metric_multiplier:.6f} params={params}'
        )

    if num_processes:
        # spawn, not fork: forking a process after jax has initialized is unsafe
        # (XLA's threads don't survive the fork), and jax may already be live in
        # the parent whenever a jax-backed rating system is swept
        with get_context('spawn').Pool(
            num_processes,
            initializer=_init_grid_search_worker,
            initargs=(rating_system_class, dataset, metrics_mask),
        ) as pool:
            for idx, current_metrics in enumerate(pool.imap(_grid_search_task, param_configurations)):
                all_metrics.append(current_metrics)
                report(idx, param_configurations[idx], current_metrics)
    else:
        for idx, params in enumerate(param_configurations):
            current_metrics = eval_wrapper(params, rating_system_class, dataset, metrics_mask)
            all_metrics.append(current_metrics)
            report(idx, params, current_metrics)
    return all_metrics


def grid_search(
    rating_system_class,
    dataset,
    param_configurations,
    metrics_mask=None,
    metric='log_loss',
    minimize_metric=True,
    num_processes=None,
    return_all_metrics=False,
    verbose=False,
):
    """
    Evaluate an explicit list of hyperparameter configurations and return the best.

    Parameters:
        rating_system_class: OnlineRatingSystem subclass to evaluate.
        dataset (TimedPairDataset): Dataset to fit and evaluate on.
        param_configurations (list[dict]): Constructor kwargs to try, one dict per configuration.
        metrics_mask (np.ndarray, optional): Boolean mask selecting which matchups count towards the metrics.
        metric (str, optional): Key of binary_metrics_suite to optimize. Defaults to 'log_loss'.
        minimize_metric (bool, optional): Whether lower is better for the metric. Defaults to True.
        num_processes (int, optional): If set, evaluate configurations in a spawn-context multiprocessing
            Pool. Single-process is usually faster: hyperparameters are traced, so sequential fits share
            one compiled scan, while every worker process recompiles from scratch.
        return_all_metrics (bool, optional): Also return the metrics of every configuration.
        verbose (bool, optional): Print progress and the running best after each configuration.

    Returns:
        (best_params, best_metrics) or (best_params, best_metrics, all_metrics).
    """
    best_params = {}
    best_metrics = {}
    metric_multiplier = 1.0 if minimize_metric else -1.0
    best_metric = np.inf

    all_metrics = _run_evals(
        rating_system_class,
        dataset,
        param_configurations,
        metrics_mask,
        metric,
        metric_multiplier,
        num_processes,
        verbose,
    )

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


def _sample_params(param_distributions, rng):
    """Draw one configuration from the random_search distribution specs."""
    params = {}
    for name, spec in param_distributions.items():
        if callable(spec):
            params[name] = spec(rng)
        elif isinstance(spec, list):
            params[name] = spec[rng.integers(len(spec))]
        elif isinstance(spec, tuple) and len(spec) == 3 and spec[0] == 'log':
            params[name] = math.exp(rng.uniform(math.log(spec[1]), math.log(spec[2])))
        elif isinstance(spec, tuple) and len(spec) == 3 and spec[0] == 'int':
            params[name] = int(rng.integers(spec[1], spec[2] + 1))
        elif isinstance(spec, tuple) and len(spec) == 2:
            params[name] = float(rng.uniform(spec[0], spec[1]))
        else:
            raise ValueError(
                f'invalid distribution for {name!r}: use (low, high) for uniform, ("log", low, high) for '
                f'log-uniform, ("int", low, high) for integers, a list for categorical, or a callable(rng)'
            )
    return params


def random_search(
    rating_system_class,
    dataset,
    param_distributions,
    num_samples=50,
    metrics_mask=None,
    metric='log_loss',
    minimize_metric=True,
    num_processes=None,
    return_all_metrics=False,
    seed=None,
    verbose=False,
):
    """
    Random search over hyperparameter distributions and return the best configuration found.

    Parameters:
        rating_system_class: OnlineRatingSystem subclass to evaluate.
        dataset (TimedPairDataset): Dataset to fit and evaluate on.
        param_distributions (dict): Maps constructor kwarg names to a sampling spec:
            (low, high) samples uniformly, ('log', low, high) log-uniformly,
            ('int', low, high) uniform integers (inclusive), a list samples a random
            element, and a callable is invoked with a numpy Generator.
        num_samples (int, optional): Number of configurations to sample. Defaults to 50.
        seed (int, optional): Seed for reproducible sampling.

    The remaining parameters and the return value match grid_search.
    """
    rng = np.random.default_rng(seed)
    param_configurations = [_sample_params(param_distributions, rng) for _ in range(num_samples)]
    return grid_search(
        rating_system_class,
        dataset,
        param_configurations,
        metrics_mask=metrics_mask,
        metric=metric,
        minimize_metric=minimize_metric,
        num_processes=num_processes,
        return_all_metrics=return_all_metrics,
        verbose=verbose,
    )


def optuna_search(
    rating_system_class,
    dataset,
    param_distributions,
    num_trials=100,
    metrics_mask=None,
    metric='log_loss',
    minimize_metric=True,
    seed=None,
    verbose=False,
    return_study=False,
):
    """
    Hyperparameter search with optuna's TPE sampler. Requires the optional optuna dependency.

    Parameters:
        rating_system_class: OnlineRatingSystem subclass to evaluate.
        dataset (TimedPairDataset): Dataset to fit and evaluate on.
        param_distributions (dict): Same specs as random_search, minus callables:
            (low, high) suggests a float, ('log', low, high) a log-scaled float,
            ('int', low, high) an integer (inclusive), and a list a categorical choice.
        num_trials (int, optional): Number of optuna trials. Defaults to 100.
        metrics_mask (np.ndarray, optional): Boolean mask selecting which matchups count towards the metrics.
        metric (str, optional): Key of binary_metrics_suite to optimize. Defaults to 'log_loss'.
        minimize_metric (bool, optional): Whether lower is better for the metric. Defaults to True.
        seed (int, optional): Seed for the TPE sampler.
        verbose (bool, optional): Keep optuna's per-trial logging and progress bar.
        return_study (bool, optional): Also return the optuna study.

    Returns:
        (best_params, best_metrics) or (best_params, best_metrics, study).
    """
    try:
        import optuna
    except ImportError as e:
        raise ImportError(
            'optuna_search requires optuna, which is an optional dependency. Install it with: uv add "riix[optuna]"'
        ) from e

    if not verbose:
        optuna.logging.set_verbosity(optuna.logging.WARNING)

    def suggest_params(trial):
        params = {}
        for name, spec in param_distributions.items():
            if isinstance(spec, list):
                params[name] = trial.suggest_categorical(name, spec)
            elif isinstance(spec, tuple) and len(spec) == 3 and spec[0] == 'log':
                params[name] = trial.suggest_float(name, spec[1], spec[2], log=True)
            elif isinstance(spec, tuple) and len(spec) == 3 and spec[0] == 'int':
                params[name] = trial.suggest_int(name, spec[1], spec[2])
            elif isinstance(spec, tuple) and len(spec) == 2:
                params[name] = trial.suggest_float(name, spec[0], spec[1])
            else:
                raise ValueError(
                    f'invalid distribution for {name!r}: use (low, high), ("log", low, high), '
                    f'("int", low, high), or a list of categorical choices'
                )
        return params

    def objective(trial):
        params = suggest_params(trial)
        current_metrics = eval_wrapper(params, rating_system_class, dataset, metrics_mask)
        trial.set_user_attr('metrics', current_metrics)
        return current_metrics[metric]

    study = optuna.create_study(
        direction='minimize' if minimize_metric else 'maximize',
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    study.optimize(objective, n_trials=num_trials, show_progress_bar=verbose)

    best_params = study.best_params
    best_metrics = study.best_trial.user_attrs['metrics']
    if return_study:
        return best_params, best_metrics, study
    return best_params, best_metrics
