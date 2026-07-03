"""
Regression tests for batched updates that used to be broken (OnlineDiscDecomp was a
copy-paste of Elo referencing attributes it doesn't have, EloMentum was a stub).
When every rating period contains a single matchup, a batched update must produce
exactly the same ratings as the online update.
"""

import numpy as np
import pytest
from riix.models.elo import Elo
from riix.models.elomentum import EloMentum
from riix.models.online_disc_decomp import OnlineDiscDecomp
from riix.utils.data_utils import TimedPairDataset


@pytest.fixture(scope='module')
def dataset():
    rng = np.random.default_rng(7)
    num_matchups = 200
    matchups = rng.integers(0, 8, size=(num_matchups, 2))
    matchups[:, 1] = (matchups[:, 0] + 1 + matchups[:, 1] % 7) % 8  # no self-matchups
    outcomes = rng.choice([0.0, 0.5, 1.0], size=num_matchups)
    time_steps = np.arange(num_matchups)  # one matchup per rating period
    return TimedPairDataset.init_from_arrays(
        time_steps=time_steps, matchups=matchups, outcomes=outcomes, competitors=list(range(8))
    )


@pytest.mark.parametrize(
    'model_class,kwargs',
    [
        (Elo, {}),
        (OnlineDiscDecomp, {}),
        (EloMentum, {'momentum_type': 'heavy_ball'}),
        (EloMentum, {'momentum_type': 'nesterov'}),
        (EloMentum, {'momentum_type': 'adam', 'momentum': (0.2, 0.9)}),
    ],
)
def test_batched_matches_online_on_single_matchup_periods(model_class, kwargs, dataset):
    online = model_class(competitors=dataset.competitors, update_method='online', **kwargs)
    batched = model_class(competitors=dataset.competitors, update_method='batched', **kwargs)
    online_probs = online.fit_dataset(dataset, return_pre_match_probs=True).probs
    batched_probs = batched.fit_dataset(dataset, return_pre_match_probs=True).probs
    np.testing.assert_allclose(online_probs, batched_probs, rtol=1e-12)
    np.testing.assert_allclose(online.get_pre_match_ratings(dataset.matchups), batched.get_pre_match_ratings(dataset.matchups), rtol=1e-12)


def test_disc_decomp_vs_actually_update(dataset):
    model = OnlineDiscDecomp(competitors=dataset.competitors, update_method='online')
    model.fit_dataset(dataset)
    assert not np.allclose(model.vs, model.initial_v)
