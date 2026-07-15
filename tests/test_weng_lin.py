"""
Known-value test for the Weng-Lin Bradley-Terry model against openskill.py.

Reference values generated with openskill 6.2.0:

    from openskill.models import BradleyTerryFull
    model = BradleyTerryFull(mu=25.0, sigma=25.0 / 3.0, beta=25.0 / 6.0, kappa=0.0001, tau=0.0)
    ratings = [model.rating() for _ in range(4)]
    for winner, loser in [(0, 1), (2, 1), (0, 2), (3, 0), (0, 1), (2, 3), (1, 3), (0, 3)]:
        [[ratings[winner]], [ratings[loser]]] = model.rate([[ratings[winner]], [ratings[loser]]])

The riix WengLin 'tm' model is intentionally not compared here: openskill's
Thurstone-Mosteller draw-margin handling differs slightly, and the TM-family math is
already covered by the TrueSkill known-value test.
"""

import pytest
import numpy as np
import jax
from riix.models.weng_lin import WengLin
from riix.utils.data_utils import TimedPairDataset


@pytest.fixture(scope='module', autouse=True)
def enable_x64():
    """Reference values need float64; jax needs the x64 flag for float64 arrays."""
    jax.config.update('jax_enable_x64', True)
    yield
    jax.config.update('jax_enable_x64', False)


OPENSKILL_MUS = [30.6175154257, 21.3151811589, 27.6123002403, 20.4856050247]
OPENSKILL_SIGMAS = [7.1963613475, 7.3888666455, 7.5770615869, 7.3518467785]


def test_weng_lin_bradley_terry_matches_openskill():
    # first index of each matchup is the winner, one matchup per rating period
    matchups = np.array([[0, 1], [2, 1], [0, 2], [3, 0], [0, 1], [2, 3], [1, 3], [0, 3]])
    outcomes = np.ones(matchups.shape[0])
    time_steps = np.arange(matchups.shape[0])
    dataset = TimedPairDataset.init_from_arrays(
        time_steps=time_steps, matchups=matchups, outcomes=outcomes, competitors=[0, 1, 2, 3]
    )
    # tau=0.0 because openskill applies tau at every rate() call while riix applies it
    # per elapsed rating period, exact params instead of the rounded riix defaults
    model = WengLin(
        competitors=dataset.competitors,
        initial_mu=25.0,
        initial_sigma=25.0 / 3.0,
        beta=25.0 / 6.0,
        kappa=0.0001,
        tau=0.0,
        draw_probability=0.0,
        model='bt',
        update_method='online',
        dtype=np.float64,
    )
    model.fit_dataset(dataset)

    np.testing.assert_allclose(model.mus, OPENSKILL_MUS, atol=1e-9)
    np.testing.assert_allclose(np.sqrt(model.sigma2s), OPENSKILL_SIGMAS, atol=1e-9)
