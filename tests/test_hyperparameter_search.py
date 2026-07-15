"""Tests for the hyperparameter search utilities in riix.eval."""

import numpy as np
import pytest

from riix.eval import _sample_params, evaluate, grid_search, optuna_search, random_search
from riix.metrics import binary_metrics_suite
from riix.models.elo import Elo
from riix.utils import generate_pair_data
from riix.utils.data_utils import TimedPairDataset


@pytest.fixture(scope='module')
def dataset():
    df = generate_pair_data(num_matchups=1000, num_competitors=25, num_rating_periods=5, seed=3)
    return TimedPairDataset(
        df,
        competitor_cols=['competitor_1', 'competitor_2'],
        outcome_col='outcome',
        datetime_col='date',
        rating_period='1D',
        verbose=False,
    )


def test_sample_params_specs():
    rng = np.random.default_rng(0)
    distributions = {
        'uniform': (1.0, 2.0),
        'log': ('log', 1e-3, 1e3),
        'int': ('int', 1, 5),
        'categorical': ['a', 'b'],
        'callable': lambda rng: rng.normal(),
    }
    for _ in range(20):
        params = _sample_params(distributions, rng)
        assert 1.0 <= params['uniform'] <= 2.0
        assert 1e-3 <= params['log'] <= 1e3
        assert isinstance(params['int'], int) and 1 <= params['int'] <= 5
        assert params['categorical'] in ('a', 'b')
        assert isinstance(params['callable'], float)
    with pytest.raises(ValueError, match='invalid distribution'):
        _sample_params({'bad': 3}, rng)


def test_random_search(dataset):
    best_params, best_metrics, all_metrics = random_search(
        Elo,
        dataset,
        param_distributions={'k': ('log', 1.0, 256.0)},
        num_samples=6,
        seed=0,
        return_all_metrics=True,
    )
    assert 1.0 <= best_params['k'] <= 256.0
    assert len(all_metrics) == 6
    assert best_metrics['log_loss'] == min(m['log_loss'] for m in all_metrics)


def test_random_search_is_seeded(dataset):
    kwargs = {'param_distributions': {'k': (1.0, 100.0)}, 'num_samples': 3, 'seed': 42}
    best_params_1, _ = random_search(Elo, dataset, **kwargs)
    best_params_2, _ = random_search(Elo, dataset, **kwargs)
    assert best_params_1 == best_params_2


def test_random_search_multiprocessing(dataset):
    best_params, best_metrics = random_search(
        Elo,
        dataset,
        param_distributions={'k': (1.0, 100.0)},
        num_samples=4,
        seed=0,
        num_processes=2,
    )
    assert 'k' in best_params
    assert 'log_loss' in best_metrics


def test_grid_search_verbose_progress(dataset, capsys):
    configurations = [{'k': 16.0}, {'k': 32.0}, {'k': 64.0}]
    grid_search(Elo, dataset, configurations, verbose=True)
    lines = [line for line in capsys.readouterr().out.splitlines() if line.startswith('[')]
    assert len(lines) == 3
    assert lines[0].startswith('[1/3] log_loss=')
    assert 'best=' in lines[-1]


def test_optuna_search(dataset):
    best_params, best_metrics, study = optuna_search(
        Elo,
        dataset,
        param_distributions={'k': ('log', 1.0, 256.0), 'initial_rating': [1000.0, 1500.0]},
        num_trials=5,
        seed=0,
        return_study=True,
    )
    assert 1.0 <= best_params['k'] <= 256.0
    assert best_params['initial_rating'] in (1000.0, 1500.0)
    assert len(study.trials) == 5
    assert best_metrics['log_loss'] == min(t.user_attrs['metrics']['log_loss'] for t in study.trials)


# ------------------------------------------------------------------------------------
# jax-specific behavior of the eval utilities


def test_evaluate_both_predict_modes(dataset):
    """evaluate() works in both prediction modes; the two modes are genuinely
    different evaluations of the same fit."""
    prequential = evaluate(Elo(competitors=dataset.competitors), dataset)
    period_start = evaluate(Elo(competitors=dataset.competitors, predict_mode='period_start'), dataset)
    assert np.isfinite(prequential['log_loss']) and np.isfinite(period_start['log_loss'])
    assert prequential['log_loss'] != period_start['log_loss']


def test_evaluate_metrics_mask(dataset):
    mask = np.zeros(len(dataset), dtype=np.bool_)
    mask[::2] = True
    metrics = evaluate(Elo(competitors=dataset.competitors), dataset, metrics_mask=mask)
    probs = Elo(competitors=dataset.competitors).fit_dataset(dataset, return_pre_match_probs=True).probs
    expected = binary_metrics_suite(probs[mask], dataset.outcomes[mask])
    assert metrics['log_loss'] == expected['log_loss']
    assert metrics['accuracy'] == expected['accuracy']


def test_grid_search_varies_predict_mode(dataset):
    configurations = [
        {'k': 32.0, 'predict_mode': 'prequential'},
        {'k': 32.0, 'predict_mode': 'period_start'},
        {'k': 64.0, 'predict_mode': 'prequential'},
    ]
    best_params, best_metrics, all_metrics = grid_search(Elo, dataset, configurations, return_all_metrics=True)
    assert len(all_metrics) == 3
    assert best_params in configurations
    assert best_metrics['log_loss'] == min(m['log_loss'] for m in all_metrics)
    # the two prediction modes at the same k are genuinely different evaluations
    assert all_metrics[0]['log_loss'] != all_metrics[1]['log_loss']


def test_random_search_log_distribution(dataset):
    best_params, best_metrics = random_search(
        Elo, dataset, param_distributions={'k': ('log', 1.0, 256.0)}, num_samples=4, seed=0
    )
    assert 1.0 <= best_params['k'] <= 256.0
    assert 'log_loss' in best_metrics


def test_optuna_search_log_distribution(dataset):
    best_params, best_metrics = optuna_search(
        Elo, dataset, param_distributions={'k': ('log', 1.0, 256.0)}, num_trials=4, seed=0
    )
    assert 1.0 <= best_params['k'] <= 256.0
    assert 'log_loss' in best_metrics


def test_sweep_reuses_one_compiled_scan(dataset):
    """The headline sweep property: hyperparameters are traced, so a sweep hits
    the jit cache after the first fit instead of recompiling per configuration."""
    from riix.core.base import _fit_scan

    evaluate(Elo(competitors=dataset.competitors), dataset)  # warm the cache for this (step fn, shape)
    cache_size = _fit_scan._cache_size()
    grid_search(Elo, dataset, [{'k': float(k)} for k in (8, 16, 32, 64)])
    assert _fit_scan._cache_size() == cache_size


def test_random_search_multiprocessing_spawn(dataset):
    """The Pool uses the spawn context, which stays safe even though jax is already
    initialized in the parent process (fork would not be)."""
    best_params, best_metrics = random_search(
        Elo,
        dataset,
        param_distributions={'k': (1.0, 100.0)},
        num_samples=2,
        seed=0,
        num_processes=2,
    )
    assert 'k' in best_params
    assert 'log_loss' in best_metrics
