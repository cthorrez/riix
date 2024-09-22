"""classes and functions for working with rating data"""

import math
import time
from typing import List
import numpy as np
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
        verbose: bool = True,
    ):
        if len(competitor_cols) != 2:
            raise ValueError('must specify exactly 2 competitor columns')
        if (bool(datetime_col) + bool(timestamp_col) + bool(time_step_col)) != 1:
            raise ValueError('must specify only one of time_step_col, datetime_col, timestamp_col')

        if time_step_col:
            self.time_steps = df[time_step_col].astype(np.int64)
        else:
            if datetime_col:
                epoch_times = pd.to_datetime(df[datetime_col]).values.astype(np.int64) // 10**9
            if timestamp_col:
                epoch_times = df[timestamp_col].values.astype(np.int64)

            first_time = epoch_times[0]
            epoch_times = epoch_times - first_time
            period_delta = int(pd.Timedelta(rating_period).total_seconds())
            self.time_steps = epoch_times // period_delta

        self.process_time_steps()

        # map competitor names/ids to integers
        self.num_matchups = len(df)
        comp_idxs, competitors = pd.factorize(pd.concat([df[competitor_cols[0]], df[competitor_cols[1]]]), sort=True)
        self.competitors = competitors.to_list()
        self.num_competitors = len(self.competitors)
        self.competitor_to_idx = {comp: idx for idx, comp in enumerate(self.competitors)}
        self.matchups = np.column_stack([comp_idxs[:self.num_matchups], comp_idxs[self.num_matchups:]])
        self.outcomes = df[outcome_col].values.astype(np.float64)

        if verbose:
            print('loaded dataset with:')
            print(f'{self.matchups.shape[0]} matchups')
            print(f'{len(self.competitors)} unique competitors')
            print(f'{self.unique_time_steps.max()} rating periods of length {rating_period}')

    def process_time_steps(self):
        self.unique_time_steps, unique_time_step_indices = np.unique(self.time_steps, return_index=True)
        self.time_step_end_idxs = np.roll(unique_time_step_indices, shift=-1)
        self.time_step_end_idxs[-1] = self.time_steps.shape[0]

    def __iter__(self):
        """iterate batches one rating period at a time"""
        period_start_idx = 0
        for time_step, period_end_idx in zip(self.unique_time_steps, self.time_step_end_idxs):
            matchups = self.matchups[period_start_idx:period_end_idx, :]
            outcomes = self.outcomes[period_start_idx:period_end_idx]
            period_start_idx = period_end_idx
            yield matchups, outcomes, time_step

    def __len__(self):
        return self.matchups.shape[0]

    def __getitem__(self, key):
        if isinstance(key, slice):
            slice_dataset = MatchupDataset.init_from_arrays(
                time_steps=self.time_steps[key],
                matchups=self.matchups[key, :],
                outcomes=self.outcomes[key],
                competitors=self.competitors,
            )
            slice_dataset.competitor_to_idx = self.competitor_to_idx
            return slice_dataset
        else:
            raise ValueError('you can only index MatchupDataset with a slice')

    @classmethod
    def init_from_arrays(cls, time_steps, matchups, outcomes, competitors):
        dataset = cls.__new__(cls)
        dataset.time_steps = time_steps
        dataset.process_time_steps()
        dataset.outcomes = outcomes
        dataset.matchups = matchups
        dataset.competitors = competitors
        dataset.num_competitors = len(competitors)
        return dataset

    @classmethod
    def load_from_npz(cls, path):
        time_steps, matchups, outcomes = np.load(path).values()
        competitors = np.unique(matchups)
        matchups = np.searchsorted(competitors, matchups)
        dataset = cls.init_from_arrays(
            time_steps=time_steps, matchups=matchups, outcomes=outcomes, competitors=competitors
        )
        print('loaded dataset with:')
        print(f'{dataset.matchups.shape[0]} matchups')
        print(f'{len(dataset.competitors)} unique competitors')
        return dataset


def split_matchup_dataset(dataset, test_fraction=0.2):
    split_idx = math.ceil(len(dataset) * (1.0 - test_fraction))
    train_dataset = dataset[:split_idx]
    test_dataset = dataset[split_idx:]
    print(f'split into train_dataset of length {len(train_dataset)} and test_dataset of length {len(test_dataset)}')
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
