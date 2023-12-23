"""classes and functions for working with rating data"""
import math
from typing import List
import numpy as np
import pandas as pd


class RatingDataset:
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
        self.idx_to_competitor = sorted(pd.unique(df[competitor_cols].astype(str).values.ravel('K')).tolist())
        self.num_competitors = len(self.idx_to_competitor)
        self.competitor_to_idx = {comp: idx for idx, comp in enumerate(self.idx_to_competitor)}
        self.matchups = df[competitor_cols].map(lambda comp: self.competitor_to_idx[str(comp)]).values.astype(np.int64)
        self.outcomes = df[outcome_col].values.astype(np.float64)

        if verbose:
            print('loaded dataset with:')
            print(f'{self.matchups.shape[0]} matchups')
            print(f'{len(self.idx_to_competitor)} unique competitors')

        if batch_size is not None:
            self.iter_fn = self.iter_by_batch
            self.num_batches = math.ceil(len(self) / self.batch_size)
            if verbose:
                print(f'{self.num_batches} batches of length {batch_size}')
        else:
            self.iter_fn = self.iter_by_rating_period
            if verbose:
                print(f'{np.max(self.time_steps)} rating periods of length {rating_period}')

    def iter_by_rating_period(self):
        """iterate batches one rating period at a time"""
        period_start_idx = 0
        for period_end_idx in self.end_time_step_idxs:
            time_step = self.time_steps[period_start_idx]
            matchups = self.matchups[period_start_idx:period_end_idx]
            outcomes = self.outcomes[period_start_idx:period_end_idx]
            period_start_idx = period_end_idx
            yield time_step, matchups, outcomes

    def iter_by_batch(self, batch_size=None):
        """iterate in fixed size batches"""
        num_batches = math.ceil(len(self) / self.batch_size)
        batch_start_idx = 0
        for _ in range(num_batches):
            batch_end_idx = batch_start_idx + self.batch_size
            time_step = self.time_steps[batch_start_idx]
            matchups = self.matchups[batch_start_idx:batch_end_idx]
            outcomes = self.outcomes[batch_start_idx:batch_end_idx]
            batch_start_idx = batch_end_idx
            yield time_step, matchups, outcomes

    def __iter__(self):
        for tup in iter(self.iter_fn()):
            yield tup

    def __len__(self):
        return self.matchups.shape[0]
