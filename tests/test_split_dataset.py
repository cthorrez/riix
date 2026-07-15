"""
P3.2: split_pair_dataset keeps its raw index split by default but gains
align_to_period=True, snapping the split back to the start of the rating period
containing it so the train -> test continuation of fit_dataset never starts
mid-period.
"""

import numpy as np
import pytest
from riix.models.glicko import Glicko
from riix.utils.data_utils import PairDataset, TimedPairDataset, split_pair_dataset


def make_dataset(time_steps):
    n = len(time_steps)
    rng = np.random.default_rng(3)
    matchups = rng.integers(0, 4, size=(n, 2))
    matchups[:, 1] = (matchups[:, 0] + 1 + matchups[:, 1] % 3) % 4
    outcomes = rng.choice([0.0, 0.5, 1.0], size=n)
    return TimedPairDataset.init_from_arrays(
        time_steps=np.array(time_steps), matchups=matchups, outcomes=outcomes, competitors=[0, 1, 2, 3]
    )


def test_split_default_is_raw_index():
    dataset = make_dataset([0, 0, 0, 1, 1, 2, 2, 2, 2, 3])
    train, test = split_pair_dataset(dataset, test_fraction=0.3)
    assert len(train) == 7 and len(test) == 3
    assert train.time_steps[-1] == test.time_steps[0] == 2  # lands mid-period


def test_split_aligned_snaps_to_period_start():
    dataset = make_dataset([0, 0, 0, 1, 1, 2, 2, 2, 2, 3])
    train, test = split_pair_dataset(dataset, test_fraction=0.3, align_to_period=True)
    assert len(train) == 5 and len(test) == 5  # raw index 7 snaps back to period 2's start (index 5)
    assert train.time_steps[-1] == 1 and test.time_steps[0] == 2


def test_aligned_split_supports_continuation():
    dataset = make_dataset([0, 0, 0, 1, 1, 2, 2, 2, 2, 3])
    model = Glicko(competitors=dataset.competitors)

    train, test = split_pair_dataset(dataset, test_fraction=0.3)
    model.fit_dataset(train)
    with pytest.raises(ValueError):  # unaligned split starts the test set mid-period
        model.fit_dataset(test)

    model = Glicko(competitors=dataset.competitors)
    train, test = split_pair_dataset(dataset, test_fraction=0.3, align_to_period=True)
    model.fit_dataset(train)
    model.fit_dataset(test)  # aligned split continues cleanly


def test_aligned_split_requires_time_steps():
    dataset = make_dataset([0, 0, 0, 1])
    plain = PairDataset.init_from_arrays(
        matchups=dataset.matchups, outcomes=dataset.outcomes, competitors=dataset.competitors
    )
    with pytest.raises(ValueError, match='time_steps'):
        split_pair_dataset(plain, align_to_period=True)


def test_aligned_split_rejects_single_period_train():
    dataset = make_dataset([0, 0, 0, 0])
    with pytest.raises(ValueError, match='first rating period'):
        split_pair_dataset(dataset, test_fraction=0.25, align_to_period=True)
