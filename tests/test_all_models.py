"""
Smoke tests that every rating system in riix.models can fit a dataset
and produce valid win probabilities. These don't check correctness of the
math (see the per-model tests for that), they catch interface drift between
the models and the OnlineRatingSystem base class.
"""

import importlib
import inspect
import pkgutil

import numpy as np
import pytest

import riix.models
from riix.core.base import OnlineRatingSystem
from riix.utils import generate_pair_data
from riix.utils.data_utils import TimedPairDataset

# models whose constructors require more than `competitors`
EXTRA_KWARGS = {
    'BaselineRatingSystem': {'mode': 'win_rate'},
}
# not real rating systems
EXCLUDED = {'Template'}


def discover_model_classes():
    classes = []
    for mod_info in pkgutil.iter_modules(riix.models.__path__):
        try:
            module = importlib.import_module(f'riix.models.{mod_info.name}')
        except ImportError as e:
            classes.append(
                pytest.param(None, id=mod_info.name, marks=pytest.mark.skip(reason=f'import failed: {e}'))
            )
            continue
        for name, obj in inspect.getmembers(module, inspect.isclass):
            if issubclass(obj, OnlineRatingSystem) and obj.__module__ == module.__name__ and name not in EXCLUDED:
                classes.append(pytest.param(obj, id=f'{mod_info.name}.{name}'))
    return classes


@pytest.fixture(scope='module')
def dataset():
    df = generate_pair_data(num_matchups=2000, num_competitors=50, num_rating_periods=8, seed=42)
    return TimedPairDataset(
        df,
        competitor_cols=['competitor_1', 'competitor_2'],
        outcome_col='outcome',
        datetime_col='date',
        rating_period='1D',
        verbose=False,
    )


@pytest.mark.parametrize('model_class', discover_model_classes())
def test_fit_and_predict(model_class, dataset):
    kwargs = EXTRA_KWARGS.get(model_class.__name__, {})
    model = model_class(competitors=dataset.competitors, **kwargs)
    probs = model.fit_dataset(dataset, return_pre_match_probs=True)
    assert probs.shape == (len(dataset),)
    assert np.all(np.isfinite(probs))
    assert np.all((probs >= 0.0) & (probs <= 1.0))
