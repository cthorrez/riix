"""
P3.2: get_pre_match_ratings / return_pre_match_ratings implemented for
BaselineRatingSystem (count modes), ConstantVarianceGlicko, and Yuksel2024,
which raised in v1 (mode='random' still raises: its predictions are fresh draws).
"""

import numpy as np
import pytest
from riix.models.baselines import BaselineRatingSystem
from riix.models.constant_variance_glicko import ConstantVarianceGlicko
from riix.models.yuksel_2024 import Yuksel2024
from riix.utils.data_utils import TimedPairDataset


@pytest.fixture(scope='module')
def dataset():
    matchups = np.array([[0, 1], [1, 2], [2, 0], [0, 1]])
    outcomes = np.array([1.0, 0.0, 0.5, 1.0])
    time_steps = np.array([0, 0, 1, 2])
    return TimedPairDataset.init_from_arrays(
        time_steps=time_steps, matchups=matchups, outcomes=outcomes, competitors=[0, 1, 2]
    )


def test_constant_variance_glicko_pre_match_ratings(dataset):
    model = ConstantVarianceGlicko(competitors=dataset.competitors)
    ratings = model.fit_dataset(dataset, return_pre_match_ratings=True).ratings
    assert ratings.shape == (4, 2)
    np.testing.assert_allclose(ratings[0], [1500.0, 1500.0])  # first match sees initial ratings
    # standalone call evaluates the fitted state
    post = model.get_pre_match_ratings(dataset.matchups[:1])
    np.testing.assert_allclose(post[0], model.ratings[dataset.matchups[0]])


def test_yuksel_pre_match_ratings(dataset):
    model = Yuksel2024(competitors=dataset.competitors)
    ratings = model.fit_dataset(dataset, return_pre_match_ratings=True).ratings
    assert ratings.shape == (4, 4)  # (rating, phi) per competitor, interleaved
    np.testing.assert_allclose(ratings[0], [1500.0, 0.0, 1500.0, 0.0])  # initial V=0 -> phi=0
    post = model.get_pre_match_ratings(dataset.matchups[:1])
    expected_phi = np.sqrt(np.maximum(model.V, 0.0) / model.W)[dataset.matchups[0]]
    np.testing.assert_allclose(post[0, [0, 2]], model.ratings[dataset.matchups[0]])
    np.testing.assert_allclose(post[0, [1, 3]], expected_phi, rtol=1e-6)


def test_baseline_count_mode_pre_match_ratings(dataset):
    model = BaselineRatingSystem(competitors=dataset.competitors, mode='win_rate', prior=2)
    ratings = model.fit_dataset(dataset, return_pre_match_ratings=True).ratings
    assert ratings.shape == (4, 2)
    np.testing.assert_allclose(ratings[0], [0.5, 0.5])  # prior of 2 with 1 virtual win
    # match 4 is competitor 0 (2 real wins... 1 win + half a draw) vs competitor 1
    wins, apps = model.wins[dataset.matchups[3]], model.appearances[dataset.matchups[3]]
    post = model.get_pre_match_ratings(dataset.matchups[3:])
    np.testing.assert_allclose(post[0], wins / apps, rtol=1e-6)


def test_baseline_random_mode_still_raises(dataset):
    model = BaselineRatingSystem(competitors=dataset.competitors, mode='random')
    with pytest.raises(NotImplementedError):
        model.get_pre_match_ratings(dataset.matchups)
    with pytest.raises(NotImplementedError):
        model.fit_dataset(dataset, return_pre_match_ratings=True)
