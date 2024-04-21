"""
example from: http://www.glicko.net/glicko/glicko2.pdf
"""
import pytest
import numpy as np
from riix.models.glicko2 import Glicko2
from riix.utils.data_utils import MatchupDataset


def test_glicko2():
    time_steps = np.array([0, 0, 0])  # all 3 matches are in the same time step
    matchups = np.array([[0, 1], [0, 2], [0, 3]])  # competitor 0 plays in 3 matchups
    outcomes = np.array([1.0, 0.0, 0.0])  # competitor 0 wins the first and loses the next 2
    dataset = MatchupDataset.init_from_arrays(
        time_steps=time_steps, matchups=matchups, outcomes=outcomes, competitors=[0, 1, 2, 3]
    )
    model = Glicko2(competitors=dataset.competitors, tau=0.5, update_method='batched')
    model.mus = (np.array([1500.0, 1400.0, 1550.0, 1700.0]) - 1500.0) / 173.7178
    model.phis = np.array([200.0, 30.0, 100.0, 300.0]) / 173.7178
    model.fit_dataset(dataset)
    # this is a really weak tolerance but alas Mr. Glickoman rounded to 4 decimal points at each step of the example
    assert model.mus[0] == pytest.approx(-0.2069, rel=5e-4)
    assert model.phis[0] == pytest.approx(0.8722, rel=1e-8)
