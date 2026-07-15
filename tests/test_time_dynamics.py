"""
P3.2 fix: a brand-new competitor's first appearance receives exactly one rating
period of time dynamics (the current period), not the global gap since the
previous period containing any matches. Consequence tested here: a newcomer's
update is invariant to how long the system existed before their debut.
"""

import numpy as np
import jax
import pytest
from riix.models.trueskill import TrueSkill
from riix.models.weng_lin import WengLin
from riix.models.skf import VSKF
from riix.models.glicko import Glicko
from riix.models.glicko2 import Glicko2
from riix.utils.data_utils import TimedPairDataset


@pytest.fixture(scope='module', autouse=True)
def enable_x64():
    """Exact invariance comparisons want float64; jax needs the x64 flag for float64 arrays."""
    jax.config.update('jax_enable_x64', True)
    yield
    jax.config.update('jax_enable_x64', False)


def fit_with_debut_gap(model_class, kwargs, update_method, gap):
    """Competitors 0/1 play in period 0; newcomers 2/3 debut `gap` periods later and play twice."""
    matchups = np.array([[0, 1], [2, 3], [3, 2]])
    outcomes = np.array([1.0, 1.0, 0.0])
    time_steps = np.array([0, gap, gap + 1])
    dataset = TimedPairDataset.init_from_arrays(
        time_steps=time_steps, matchups=matchups, outcomes=outcomes, competitors=[0, 1, 2, 3]
    )
    model = model_class(
        competitors=dataset.competitors, update_method=update_method, dtype=np.float64, **kwargs
    )
    probs = model.fit_dataset(dataset, return_pre_match_probs=True).probs
    return model, probs


@pytest.mark.parametrize('update_method', ['online', 'batched'])
@pytest.mark.parametrize(
    'model_class,kwargs,state_attrs',
    [
        (TrueSkill, {}, ('mus', 'sigma2s')),
        (WengLin, {}, ('mus', 'sigma2s')),
        (VSKF, {'beta': 0.98, 'epsilon': 0.02}, ('mus', 'vs')),
        (Glicko, {}, ('ratings', 'rating_devs')),
        (Glicko2, {}, ('mus', 'phis', 'sigmas')),
    ],
)
def test_newcomer_update_invariant_to_debut_gap(model_class, kwargs, state_attrs, update_method):
    model_late, probs_late = fit_with_debut_gap(model_class, kwargs, update_method, gap=9)
    model_early, probs_early = fit_with_debut_gap(model_class, kwargs, update_method, gap=1)

    # the newcomers' matches play out identically regardless of the debut gap
    np.testing.assert_allclose(probs_late[1:], probs_early[1:], rtol=1e-12)
    for attr in state_attrs:
        late = getattr(model_late, attr)[2:]
        early = getattr(model_early, attr)[2:]
        np.testing.assert_allclose(late, early, rtol=1e-12, err_msg=f'{attr} differs with debut gap')
