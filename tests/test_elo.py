"""
Known-value test for Elo against independently computed classic Elo arithmetic:
    expected = 1 / (1 + 10 ** (-(r_1 - r_2) / 400))
    r_1' = r_1 + k * (outcome - expected)
"""

import pytest
import numpy as np
from riix.models.elo import Elo
from riix.utils.data_utils import TimedPairDataset


def test_elo_known_values():
    # a=1500, b=1400, c=1550 with k=32, matches played sequentially:
    # a beats b, c beats a, b draws c, a beats b
    time_steps = np.array([0, 0, 0, 0])
    matchups = np.array([[0, 1], [0, 2], [1, 2], [0, 1]])
    outcomes = np.array([1.0, 0.0, 0.5, 1.0])
    dataset = TimedPairDataset.init_from_arrays(
        time_steps=time_steps, matchups=matchups, outcomes=outcomes, competitors=['a', 'b', 'c']
    )
    model = Elo(competitors=dataset.competitors, k=32.0, update_method='online')
    model.ratings = np.array([1500.0, 1400.0, 1550.0])

    probs = model.fit_dataset(dataset, return_pre_match_probs=True).probs

    # first pre-match prob: 1 / (1 + 10 ** (-(1500 - 1400) / 400))
    assert probs[0] == pytest.approx(0.6400649998, abs=1e-9)
    assert model.ratings[0] == pytest.approx(1508.7442676924836, abs=1e-9)
    assert model.ratings[1] == pytest.approx(1384.4880691733106, abs=1e-9)
    assert model.ratings[2] == pytest.approx(1556.7676631342058, abs=1e-9)
