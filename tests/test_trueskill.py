"""
example from: https://github.com/sublee/trueskill/blob/master/trueskilltest.py
plus reference-formula tests for the P3.2 draw-moment fixes and the
variance_increase_method='match' port
"""

import math
import pytest
import numpy as np
import jax
from scipy.stats import norm as scipy_norm
from riix.models.trueskill import TrueSkill
from riix.utils.data_utils import TimedPairDataset


@pytest.fixture(scope='module', autouse=True)
def enable_x64():
    """Reference values need float64; jax needs the x64 flag for float64 arrays."""
    jax.config.update('jax_enable_x64', True)
    yield
    jax.config.update('jax_enable_x64', False)


def general_trueskill(update_method):
    time_steps = np.array([0, 0])
    matchups = np.array([[0, 1], [2, 3]])
    outcomes = np.array([1.0, 0.5])
    dataset = TimedPairDataset.init_from_arrays(
        time_steps=time_steps, matchups=matchups, outcomes=outcomes, competitors=[0, 1, 2, 3]
    )
    model = TrueSkill(
        competitors=dataset.competitors,
        initial_mu=25.0,
        initial_sigma=25.0 / 3.0,
        beta=25.0 / 6.0,
        draw_probability=0.1,
        update_method=update_method,
        dtype=np.float64,
    )
    model.fit_dataset(dataset)
    return model


def test_trueskill_online():
    model = general_trueskill('online')
    assert model.mus[0] == pytest.approx(29.396, rel=1e-4)
    assert model.mus[1] == pytest.approx(20.604, rel=1e-4)
    assert math.sqrt(model.sigma2s[0]) == pytest.approx(7.171, rel=1e-4)
    assert math.sqrt(model.sigma2s[1]) == pytest.approx(7.171, rel=1e-4)

    assert model.mus[2] == pytest.approx(25.0, rel=1e-4)
    assert model.mus[3] == pytest.approx(25.0, rel=1e-4)
    assert math.sqrt(model.sigma2s[2]) == pytest.approx(6.458, rel=5e-4)
    assert math.sqrt(model.sigma2s[3]) == pytest.approx(6.458, rel=5e-4)


def test_trueskill_batched():
    model = general_trueskill('batched')
    assert model.mus[0] == pytest.approx(29.396, rel=1e-4)
    assert model.mus[1] == pytest.approx(20.604, rel=1e-4)
    assert math.sqrt(model.sigma2s[0]) == pytest.approx(7.171, rel=1e-4)
    assert math.sqrt(model.sigma2s[1]) == pytest.approx(7.171, rel=1e-4)

    assert model.mus[2] == pytest.approx(25.0, rel=1e-4)
    assert model.mus[3] == pytest.approx(25.0, rel=1e-4)
    assert math.sqrt(model.sigma2s[2]) == pytest.approx(6.458, rel=5e-4)
    assert math.sqrt(model.sigma2s[3]) == pytest.approx(6.458, rel=5e-4)


# ------------------------------------------------------------------ P3.2 fixes

BETA = 25.0 / 6.0
DRAW_PROBABILITY = 0.10
EPSILON = scipy_norm.ppf((DRAW_PROBABILITY + 1.0) / 2.0) * math.sqrt(2.0) * BETA


def reference_win_update(mu, sigma2, outcome):
    """Two-player win/loss update per the reference TrueSkill (moserware/sublee) formulas."""
    winner, loser = (0, 1) if outcome == 1.0 else (1, 0)
    c2 = 2.0 * BETA**2 + sigma2[0] + sigma2[1]
    c = math.sqrt(c2)
    t = (mu[winner] - mu[loser]) / c
    d = t - EPSILON / c
    v = scipy_norm.pdf(d) / scipy_norm.cdf(d)
    w = v * (v + d)
    mu = list(mu)
    mu[winner] += sigma2[winner] / c * v
    mu[loser] -= sigma2[loser] / c * v
    return mu, [s2 - s2**2 / c2 * w for s2 in sigma2]


def reference_draw_update(mu, sigma2):
    """Two-player draw update per the reference TrueSkill (moserware/sublee) formulas:
    v_draw is odd in the rating difference (the leader loses mu) and w_draw is positive."""
    c2 = 2.0 * BETA**2 + sigma2[0] + sigma2[1]
    c = math.sqrt(c2)
    t = (mu[0] - mu[1]) / c
    eps_c = EPSILON / c
    a = eps_c - abs(t)
    b = -eps_c - abs(t)
    denom = scipy_norm.cdf(a) - scipy_norm.cdf(b)
    v = math.copysign(1.0, t) * (scipy_norm.pdf(b) - scipy_norm.pdf(a)) / denom
    w = (a * scipy_norm.pdf(a) - b * scipy_norm.pdf(b)) / denom + v**2
    mu_new = [mu[0] + sigma2[0] / c * v, mu[1] - sigma2[1] / c * v]
    return mu_new, [s2 - s2**2 / c2 * w for s2 in sigma2]


def fit_win_then_draw(update_method, matchup_2):
    """0 beats 1, then an asymmetric draw between `matchup_2` (one match per period, tau=0)."""
    matchups = np.array([[0, 1], matchup_2])
    dataset = TimedPairDataset.init_from_arrays(
        time_steps=np.array([0, 1]), matchups=matchups, outcomes=np.array([1.0, 0.5]), competitors=[0, 1]
    )
    model = TrueSkill(
        competitors=dataset.competitors,
        initial_mu=25.0,
        initial_sigma=25.0 / 3.0,
        beta=BETA,
        tau=0.0,
        draw_probability=DRAW_PROBABILITY,
        update_method=update_method,
        dtype=np.float64,
    )
    model.fit_dataset(dataset)
    return model


@pytest.mark.parametrize('update_method', ['online', 'batched'])
@pytest.mark.parametrize('matchup_2', [[0, 1], [1, 0]])
def test_trueskill_asymmetric_draw_matches_reference(update_method, matchup_2):
    """P3.2: draws use the reference moments in both update methods (v1's online draw
    updates were scaled by the 0.5 outcome with a sign-flipped v_draw; its batched
    vector functions dropped the draw sign entirely)."""
    model = fit_win_then_draw(update_method, matchup_2)

    mu, sigma2 = reference_win_update([25.0, 25.0], [(25.0 / 3.0) ** 2] * 2, 1.0)
    if matchup_2 == [1, 0]:
        mu, sigma2 = mu[::-1], sigma2[::-1]
    mu, sigma2 = reference_draw_update(mu, sigma2)
    if matchup_2 == [1, 0]:
        mu, sigma2 = mu[::-1], sigma2[::-1]

    np.testing.assert_allclose(model.mus, mu, rtol=1e-12)
    np.testing.assert_allclose(model.sigma2s, sigma2, rtol=1e-12)


def test_trueskill_draw_moves_leader_down():
    """A draw is bad news for the leader and good news for the trailer, whichever slot they sit in."""
    for matchup_2 in ([0, 1], [1, 0]):
        model = fit_win_then_draw('online', matchup_2)
        assert model.mus[0] < 29.396  # leader (won match 1) loses mu on the draw
        assert model.mus[1] > 20.604  # trailer gains mu
        assert np.all(model.sigma2s < 7.171**2)  # both sigmas keep shrinking


def test_trueskill_variance_increase_match():
    """P3.2 port of variance_increase_method='match': tau^2 is added to each participant
    before every match, and rating-period gaps add nothing."""
    tau = 0.5
    matchups = np.array([[0, 1], [0, 1]])
    outcomes = np.array([1.0, 0.5])

    mu, sigma2 = [25.0, 25.0], [(25.0 / 3.0) ** 2] * 2
    sigma2 = [s2 + tau**2 for s2 in sigma2]
    mu, sigma2 = reference_win_update(mu, sigma2, 1.0)
    sigma2 = [s2 + tau**2 for s2 in sigma2]
    mu, sigma2 = reference_draw_update(mu, sigma2)

    for update_method, time_steps in [('online', [0, 1]), ('online', [0, 9]), ('batched', [0, 9])]:
        dataset = TimedPairDataset.init_from_arrays(
            time_steps=np.array(time_steps), matchups=matchups, outcomes=outcomes, competitors=[0, 1]
        )
        model = TrueSkill(
            competitors=dataset.competitors,
            initial_mu=25.0,
            initial_sigma=25.0 / 3.0,
            beta=BETA,
            tau=tau,
            draw_probability=DRAW_PROBABILITY,
            variance_increase_method='match',
            update_method=update_method,
            dtype=np.float64,
        )
        model.fit_dataset(dataset)
        np.testing.assert_allclose(model.mus, mu, rtol=1e-12)
        np.testing.assert_allclose(model.sigma2s, sigma2, rtol=1e-12)


def test_trueskill_defaults():
    """P3.2 decision: TrueSkill defaults to update_method='online' like every other
    system (v1 defaulted to 'batched') and variance_increase_method='time'."""
    model = TrueSkill(competitors=[0, 1])
    assert model.update_method == 'online'
    assert model.variance_increase_method == 'time'
