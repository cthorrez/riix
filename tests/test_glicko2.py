"""
example from: http://www.glicko.net/glicko/glicko2.pdf
"""

import pytest
import numpy as np
import jax
from riix.models.glicko2 import Glicko2
from riix.utils.data_utils import TimedPairDataset


@pytest.fixture(scope='module', autouse=True)
def enable_x64():
    """Reference values need float64; jax needs the x64 flag for float64 arrays."""
    jax.config.update('jax_enable_x64', True)
    yield
    jax.config.update('jax_enable_x64', False)


def test_glicko2():
    time_steps = np.array([0, 0, 0])  # all 3 matches are in the same time step
    matchups = np.array([[0, 1], [0, 2], [0, 3]])  # competitor 0 plays in 3 matchups
    outcomes = np.array([1.0, 0.0, 0.0])  # competitor 0 wins the first and loses the next 2
    dataset = TimedPairDataset.init_from_arrays(
        time_steps=time_steps, matchups=matchups, outcomes=outcomes, competitors=[0, 1, 2, 3]
    )
    model = Glicko2(competitors=dataset.competitors, tau=0.5, update_method='batched', dtype=np.float64)
    model.mus = (np.array([1500.0, 1400.0, 1550.0, 1700.0]) - 1500.0) / 173.7178
    model.phis = np.array([200.0, 30.0, 100.0, 300.0]) / 173.7178
    model.fit_dataset(dataset)
    # this is a really weak tolerance but alas Mr. Glickoman rounded to 4 decimal points at each step of the example
    assert model.mus[0] == pytest.approx(-0.2069, rel=5e-4)
    assert model.phis[0] == pytest.approx(0.8722, rel=5e-4)
    assert model.sigmas[0] == pytest.approx(0.05999, rel=5e-4)


def test_glicko2_batched_phi_star_capped():
    """P3.2 fix: batched phi_star is capped at the initial phi like the online path
    (v1 left it uncapped, so fresh competitors exceeded the initial deviation).

    Two fresh competitors sit exactly at the cap, so phi_star^2 must equal max_phi2
    and the update becomes independent of the solved volatility — the expected
    posterior is computable in closed form.
    """
    dataset = TimedPairDataset.init_from_arrays(
        time_steps=np.array([0]), matchups=np.array([[0, 1]]), outcomes=np.array([1.0]), competitors=[0, 1]
    )
    model = Glicko2(competitors=dataset.competitors, update_method='batched', dtype=np.float64)
    model.fit_dataset(dataset)

    max_phi2 = (350.0 / 173.7178) ** 2
    g = 1.0 / np.sqrt(1.0 + 3.0 * max_phi2 / np.pi**2)
    expected_score = 0.5  # equal ratings
    v = 1.0 / (g**2 * expected_score * (1.0 - expected_score))
    expected_phi2 = 1.0 / (1.0 / max_phi2 + 1.0 / v)  # phi_star^2 capped to exactly max_phi2
    expected_mu = expected_phi2 * g * (1.0 - expected_score)

    assert model.phis[0] ** 2 == pytest.approx(expected_phi2, rel=1e-10)
    assert model.phis[1] ** 2 == pytest.approx(expected_phi2, rel=1e-10)
    assert model.mus[0] == pytest.approx(expected_mu, rel=1e-10)
    assert model.mus[1] == pytest.approx(-expected_mu, rel=1e-10)
