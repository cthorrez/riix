"""classes and functions for working with rating data"""

import math
import time
from typing import List
from copy import deepcopy
import numpy as np
import scipy
import pandas as pd


class MatchupDataset:
    """class for loading and iterating over paired comparison data"""

    def __init__(
        self,
        df: pd.DataFrame,
        competitor_cols: List[str],
        outcome_col: str,
        datetime_col: str = None,
        timestamp_col: str = None,
        time_step_col: str = None,
        rating_period: str = '1W',
        batch_size: int = None,
        verbose: bool = True,
    ):
        if len(competitor_cols) != 2:
            raise ValueError('must specify exactly 2 competitor columns')
        if (bool(datetime_col) + bool(timestamp_col) + bool(time_step_col)) != 1:
            raise ValueError('must specify only one of time_step_col, datetime_col, timestamp_col')

        self.batch_size = batch_size

        if time_step_col:
            self.time_steps = df[time_step_col].astype(np.int64)
        else:
            if datetime_col:
                # get integer time_steps starting from 0 and increasing one per rating period
                epoch_times = pd.to_datetime(df[datetime_col]).values.astype(np.int64) // 10**9
            if timestamp_col:
                epoch_times = df[timestamp_col].values.astype(np.int64)

            first_time = epoch_times[0]
            epoch_times = epoch_times - first_time
            period_delta = int(pd.Timedelta(rating_period).total_seconds())
            self.time_steps = epoch_times // period_delta

        # _, self.time_step_start_idxs, time_step_counts = np.unique(
        #     self.time_steps, return_index=True, return_counts=True
        # )
        # self.time_step_end_idxs = self.time_step_start_idxs + time_step_counts
        self.unique_time_steps = np.unique(self.time_steps)
        self.time_slices = scipy.ndimage.find_objects(
            self.time_steps + 1
        )  # LOL! https://codereview.stackexchange.com/a/60887

        # map competitor names/ids to integers
        self.competitors = sorted(pd.unique(df[competitor_cols].astype(str).values.ravel('K')).tolist())
        self.num_competitors = len(self.competitors)
        self.competitor_to_idx = {comp: idx for idx, comp in enumerate(self.competitors)}
        self.matchups = df[competitor_cols].map(lambda comp: self.competitor_to_idx[str(comp)]).values.astype(np.int64)
        self.outcomes = df[outcome_col].values.astype(np.float64)

        if verbose:
            print('loaded dataset with:')
            print(f'{self.matchups.shape[0]} matchups')
            print(f'{len(self.competitors)} unique competitors')

        if batch_size is not None:
            self.iter_fn = self.iter_by_batch
            self.num_batches = math.ceil(len(self) / self.batch_size)
            if verbose:
                print(f'{self.num_batches} batches of length {batch_size}')
        else:
            self.iter_fn = self.iter_by_rating_period
            if verbose and (time_step_col is None):
                print(f'{np.max(self.time_steps) + 1} rating periods of length {rating_period}')

    def iter_by_rating_period(self):
        """iterate batches one rating period at a time"""
        for time_step in np.nditer(op=self.unique_time_steps):
            idx_slice = self.time_slices[time_step]
            matchups = self.matchups[idx_slice]
            outcomes = self.outcomes[idx_slice]
            yield matchups, outcomes, time_step

    def iter_by_batch(self, batch_size=None):
        """iterate in fixed size batches"""
        num_batches = math.ceil(len(self) / self.batch_size)
        batch_start_idx = 0
        for _ in range(num_batches):
            batch_end_idx = batch_start_idx + self.batch_size
            time_step = self.time_steps[
                batch_start_idx
            ]  # it's possible the batch represents data from multiple time_steps
            matchups = self.matchups[batch_start_idx:batch_end_idx]
            outcomes = self.outcomes[batch_start_idx:batch_end_idx]
            batch_start_idx = batch_end_idx
            yield matchups, outcomes, time_step

    def __iter__(self):
        for batch in iter(self.iter_fn()):
            yield batch

    def __len__(self):
        return self.matchups.shape[0]

    @classmethod
    def load_from_npz(cls, path):
        dataset = cls.__new__(cls)
        time_steps, matchups, outcomes = np.load(path).values()
        dataset.time_steps = time_steps
        dataset.unique_time_steps = np.unique(dataset.time_steps)
        dataset.time_slices = scipy.ndimage.find_objects(dataset.time_steps + 1)
        dataset.outcomes = outcomes
        dataset.competitors = np.unique(matchups)
        dataset.matchups = np.searchsorted(dataset.competitors, matchups)
        dataset.num_competitors = dataset.competitors.shape[0]
        dataset.iter_fn = dataset.iter_by_rating_period
        return dataset


def split_matchup_dataset(dataset, test_fraction=0.2):
    train_dataset = deepcopy(dataset)
    test_dataset = deepcopy(dataset)

    num_matchups = len(dataset)
    num_train_matchups = math.ceil(num_matchups * (1.0 - test_fraction))

    train_dataset.matchups = dataset.matchups[:num_train_matchups, :]
    train_dataset.outcomes = dataset.outcomes[:num_train_matchups]
    train_dataset.time_steps = dataset.time_steps[:num_train_matchups]
    # _, train_dataset.time_step_start_idxs, train_time_step_counts = np.unique(
    #     train_dataset.time_steps, return_index=True, return_counts=True
    # )
    # train_dataset.time_step_end_idxs = train_dataset.time_step_start_idxs + train_time_step_counts
    train_dataset.unique_time_steps = np.unique(train_dataset.time_steps)
    train_dataset.time_slices = scipy.ndimage.find_objects(train_dataset.time_steps + 1)

    test_dataset.matchups = dataset.matchups[num_train_matchups:, :]
    test_dataset.outcomes = dataset.outcomes[num_train_matchups:]
    test_dataset.time_steps = dataset.time_steps[num_train_matchups:]
    # _, test_dataset.time_step_start_idxs, test_time_step_counts = np.unique(
    #     test_dataset.time_steps, return_index=True, return_counts=True
    # )
    # test_dataset.time_step_end_idxs = test_dataset.time_step_start_idxs + test_time_step_counts
    test_dataset.unique_time_steps = np.unique(test_dataset.time_steps)
    test_dataset.time_slices = scipy.ndimage.find_objects(test_dataset.time_steps + 1)

    return train_dataset, test_dataset


def generate_matchup_data(
    num_matchups: int = 10000,
    num_competitors: int = 100,
    num_rating_periods: int = 10,
    strength_var: float = 1.0,
    strength_noise_var: float = 0.1,
    theta: float = 1.001,
    seed: int = 0,
):
    start_time = int(time.time())
    matchups_per_period = num_matchups // num_rating_periods
    period_offsets = np.arange(num_rating_periods) * (3600 * 24)
    initial_timestamps = np.zeros(num_rating_periods, dtype=np.int64) + start_time
    timestamps = (initial_timestamps + period_offsets).repeat(matchups_per_period)

    rng = np.random.default_rng(seed=seed)
    strength_means = rng.normal(loc=0.0, scale=math.sqrt(strength_var), size=num_competitors)
    comp_1 = rng.integers(low=0, high=num_competitors, size=(num_matchups, 1))
    offset = rng.integers(low=1, high=num_competitors, size=(num_matchups, 1))
    comp_2 = np.mod(comp_1 + offset, num_competitors)
    matchups = np.hstack([comp_1, comp_2])
    strengths = strength_means[matchups]

    strength_noise = rng.normal(loc=0.0, scale=math.sqrt(strength_noise_var), size=strengths.shape)
    strengths = strengths + strength_noise
    # they need to be positive!
    strengths = np.exp(strengths)

    probs = np.zeros(shape=(num_matchups, 3))  # p(comp_1 win), p(draw), p(comp_2 win)
    probs[:, 0] = strengths[:, 0] / (strengths[:, 0] + (theta * strengths[:, 1]))
    probs[:, 2] = strengths[:, 1] / (strengths[:, 1] + (theta * strengths[:, 0]))
    probs[:, 1] = 1.0 - probs[:, 0] - probs[:, 2]

    outcomes = rng.multinomial(n=1, pvals=probs)
    outcomes = np.argmax(outcomes, axis=1) / 2.0  # map 0->0, 1->0.5, 2->1.0

    data = {
        'timestamp': timestamps,
        'competitor_1': matchups[:, 0],
        'competitor_2': matchups[:, 1],
        'outcome': outcomes,
    }
    df = pd.DataFrame(data)
    df['date'] = pd.to_datetime(df['timestamp'], unit='s')
    df['competitor_1'] = 'competitor_' + df['competitor_1'].astype(str)
    df['competitor_2'] = 'competitor_' + df['competitor_2'].astype(str)
    return df
