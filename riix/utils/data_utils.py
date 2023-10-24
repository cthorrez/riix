"""classes and functions for working with rating data"""
from typing import List
import numpy as np
import pandas as pd

class RatingDataset:
    """class for loading and iterating over paired comparison data"""
    def __init__(
        self,
        df:pd.DataFrame,
        competitor_cols:List[str],
        outcome_col:str,
        datetime_col:str,
        rating_period:str='1W'
    ):
        if len(competitor_cols) != 2:
            raise ValueError('must specify exactly 2 competitor columns')

        # get integer time_steps starting from 0 and increasing one per rating period
        epoch_times = pd.to_datetime(df[datetime_col]).values.astype(np.int64) // 10 ** 9
        first_time = epoch_times[0]
        epoch_times = epoch_times - first_time
        period_delta = int(pd.Timedelta(rating_period).total_seconds())
        self.time_steps = epoch_times // period_delta
        _, self.time_step_idxs = np.unique(self.time_steps, return_index=True)

        del epoch_times, first_time, period_delta # free up memory before moving on

        # map competitor names/ids to integers
        self.idx_to_competitor = sorted(pd.unique(df[competitor_cols].astype(str).values.ravel('K')).tolist())
        competitor_to_idx = {comp : idx for idx,comp in enumerate(self.idx_to_competitor)}
        self.matchups = df[competitor_cols].map(lambda comp : competitor_to_idx[str(comp)]).values.astype(np.int64)
        self.outcomes = df[outcome_col].values.astype(np.float64)

        print('loaded dataset with:')
        print(f'{self.matchups.shape[0]} matchups')
        print(f'{len(self.idx_to_competitor)} unique competitors')
        print(f'{np.max(self.time_steps)} rating periods of length {rating_period}')

    def __iter__(self):
        period_start_idx = 0
        for period_end_idx in self.time_step_idxs[1:-1]:
            time_step = self.time_steps[period_start_idx]
            matchups = self.matchups[period_start_idx:period_end_idx]
            outcomes = self.outcomes[period_start_idx:period_end_idx]
            period_start_idx = period_end_idx
            yield time_step, matchups, outcomes

