"""
example from: https://github.com/sublee/trueskill/blob/master/trueskilltest.py
"""
import math
import pytest
import numpy as np
from riix.models.trueskill import TrueSkill
from riix.utils.data_utils import MatchupDataset


def general_trueskill(update_method):
    time_steps = np.array([0, 0])
    matchups = np.array([[0, 1], [2, 3]])
    outcomes = np.array([1.0, 0.5])
    dataset = MatchupDataset.init_from_arrays(
        time_steps=time_steps, matchups=matchups, outcomes=outcomes, competitors=[0, 1, 2, 3]
    )
    model = TrueSkill(
        competitors=dataset.competitors,
        initial_mu=25.0,
        initial_sigma=25.0 / 3.0,
        beta=25.0 / 6.0,
        draw_probability=0.1,
        update_method=update_method,
    )
    model.fit_dataset(dataset)
    return model


def test_trueskill_iterative():
    model = general_trueskill('iterative')
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
