"""classes and functions for working with rating data"""

import math
import time
from typing import Optional, List, NamedTuple
import numpy as np
from scipy.special import expit as sigmoid
import pandas as pd


class MatchupBatch(NamedTuple):
    """class to hold a batch of matchupdata"""

    matchups: np.ndarray
    outcomes: np.ndarray
    time_step: Optional[int] = None


class MatchupDataset:
    """class for loading and iterating over paired comparison data"""

    def __init__(
        self,
        df: pd.DataFrame,
        competitor_cols: List[str],
        outcome_col: str,
        datetime_col: str = None,
        timestamp_col: str = None,
        rating_period: str = '1W',
        batch_size: int = None,
        verbose: bool = True,
    ):
        if len(competitor_cols) != 2:
            raise ValueError('must specify exactly 2 competitor columns')
        if (bool(datetime_col) + bool(timestamp_col)) != 1:
            raise ValueError('must specify only one of datetime_col and timestamp_col')

        self.batch_size = batch_size

        if datetime_col:
            # get integer time_steps starting from 0 and increasing one per rating period
            epoch_times = pd.to_datetime(df[datetime_col]).values.astype(np.int64) // 10**9
        if timestamp_col:
            epoch_times = df[timestamp_col].values.astype(np.int64)

        first_time = epoch_times[0]
        epoch_times = epoch_times - first_time
        period_delta = int(pd.Timedelta(rating_period).total_seconds())
        self.time_steps = epoch_times // period_delta
        _, start_time_step_idxs = np.unique(self.time_steps, return_index=True)
        self.end_time_step_idxs = np.append(start_time_step_idxs[1:], self.time_steps.shape[0])

        del epoch_times, first_time, period_delta  # free up memory before moving on

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
            if verbose:
                print(f'{np.max(self.time_steps) + 1} rating periods of length {rating_period}')

    def iter_by_rating_period(self) -> MatchupBatch:
        """iterate batches one rating period at a time"""
        period_start_idx = 0
        for period_end_idx in self.end_time_step_idxs:
            time_step = self.time_steps[period_start_idx]
            matchups = self.matchups[period_start_idx:period_end_idx]
            outcomes = self.outcomes[period_start_idx:period_end_idx]
            period_start_idx = period_end_idx
            yield MatchupBatch(matchups, outcomes, time_step)

    def iter_by_batch(self, batch_size=None) -> MatchupBatch:
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
            yield MatchupBatch(matchups, outcomes, time_step)

    def __iter__(self):
        for batch in iter(self.iter_fn()):
            yield batch

    def __len__(self):
        return self.matchups.shape[0]


def generate_matchup_data(
    num_matchups: int = 1000,
    num_competitors: int = 100,
    num_rating_periods: int = 10,
    skill_var: float = 1.0,
    outcome_noise_var: float = 0.1,
    draw_margin: float = 0.1,
    seed: int = 0,
):
    start_time = int(time.time())
    matchups_per_period = num_matchups // num_rating_periods
    period_offsets = np.arange(num_rating_periods) * (3600 * 24)
    initial_timestamps = np.zeros(num_rating_periods, dtype=np.int64) + start_time
    timestamps = (initial_timestamps + period_offsets).repeat(matchups_per_period)

    rng = np.random.default_rng(seed=seed)
    skill_means = rng.normal(loc=0.0, scale=skill_var, size=num_competitors)
    comp_1 = rng.integers(low=0, high=num_competitors, size=(num_matchups, 1))
    offset = rng.integers(low=1, high=num_competitors, size=(num_matchups, 1))
    comp_2 = np.mod(comp_1 + offset, num_competitors)
    matchups = np.hstack([comp_1, comp_2])
    skills = skill_means[matchups]
    skill_diffs = skills[:, 0] - skills[:, 1]
    noise = rng.normal(loc=0.0, scale=outcome_noise_var)
    probs = sigmoid(skill_diffs + noise)
    outcomes = np.zeros(num_matchups) + 0.5
    outcomes[probs >= 0.5 + draw_margin] = 1.0
    outcomes[probs < 0.5 - draw_margin] = 0.0

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
