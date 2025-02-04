"""Classes and functions for working with rating data"""

import math
import time
from typing import List, Optional
import numpy as np
import polars as pl
from riix.utils.date_utils import get_duration


class PairDataset:
    """Base class for paired comparison datasets without temporal information."""

    def __init__(
        self,
        df: pl.DataFrame,
        competitor_cols: List[str],
        outcome_col: str,
        verbose: bool = True,
    ):
        self._init_competitors(df, competitor_cols)
        self._init_matchups(df, competitor_cols)
        self.outcomes = df[outcome_col].to_numpy()
        
        if verbose:
            self._print_stats()

    def _init_competitors(self, df: pl.DataFrame, competitor_cols: List[str]):
        """Initialize competitor metadata."""
        competitor_series = pl.concat([df[col].cast(pl.Utf8) for col in competitor_cols])
        self.competitors = sorted(competitor_series.unique().to_list())
        self.num_competitors = len(self.competitors)
        self.competitor_to_idx = dict(zip(self.competitors, range(self.num_competitors)))

    def _init_matchups(self, df: pl.DataFrame, competitor_cols: List[str]):
        """Create numerical matchup indices."""
        competitors_df = pl.DataFrame({'competitor': self.competitors}).lazy()
        
        matchups_df = (
            df.lazy()
            .select([
                pl.col(competitor_cols[0]).cast(pl.Utf8).alias('comp1'),
                pl.col(competitor_cols[1]).cast(pl.Utf8).alias('comp2')
            ])
            .join(
                competitors_df.with_columns(pl.int_range(pl.len(), dtype=pl.Int32).alias('index1')),
                left_on='comp1',
                right_on='competitor'
            )
            .join(
                competitors_df.with_columns(pl.int_range(pl.len(), dtype=pl.Int32).alias('index2')),
                left_on='comp2',
                right_on='competitor'
            )
            .select(['index1', 'index2'])
        )
        self.matchups = np.ascontiguousarray(matchups_df.collect().to_numpy())

    def _print_stats(self):
        """Print dataset statistics."""
        print('Loaded dataset with:')
        print(f'{len(self)} matchups')
        print(f'{self.num_competitors} unique competitors')

    def __len__(self):
        return self.matchups.shape[0]

    def __getitem__(self, key):
        if isinstance(key, slice):
            return self._create_slice(key)
        raise ValueError('Only slice indexing supported')

    def _create_slice(self, key: slice):
        """Create sliced dataset instance."""
        slice_data = {
            'matchups': self.matchups[key],
            'outcomes': self.outcomes[key],
            'competitors': self.competitors,
        }
        return self.init_from_arrays(**slice_data)

    @classmethod
    def init_from_arrays(cls, matchups: np.ndarray, outcomes: np.ndarray, competitors: list):
        """Factory method for creating datasets from arrays."""
        dataset = cls.__new__(cls)
        dataset.matchups = matchups
        dataset.outcomes = outcomes
        dataset.competitors = competitors
        dataset.num_competitors = len(competitors)
        dataset.competitor_to_idx = dict(zip(competitors, range(len(competitors))))
        return dataset


class TimedPairDataset(PairDataset):
    """Dataset for temporal paired comparisons with rating periods."""

    def __init__(
        self,
        df: pl.DataFrame,
        competitor_cols: List[str],
        outcome_col: str,
        datetime_col: Optional[str] = None,
        time_step_col: Optional[str] = None,
        rating_period: str = '1W',
        verbose: bool = True,
    ):
        super().__init__(df, competitor_cols, outcome_col, verbose=False)
        self._init_time_steps(df, datetime_col, time_step_col, rating_period)
        
        if verbose:
            print(f'{self.unique_time_steps.max() + 1} rating periods of length {rating_period}')

    def _init_time_steps(self, df: pl.DataFrame, datetime_col: Optional[str], 
                        time_step_col: Optional[str], rating_period: str):
        """Initialize temporal components."""
        if sum([bool(datetime_col), bool(time_step_col)]) != 1:
            raise ValueError('Specify exactly one of datetime_col or time_step_col')

        if time_step_col:
            self.time_steps = df[time_step_col].to_numpy()
        else:
            self.time_steps = self._convert_datetime(df[datetime_col], rating_period)

        self._process_time_steps()

    def _convert_datetime(self, datetime_series: pl.Series, rating_period: str) -> np.ndarray:
        """Convert datetime column to time steps."""
        if datetime_series.dtype == pl.Date:
            datetime_series = datetime_series.cast(pl.Datetime)
        elif datetime_series.dtype == pl.Utf8:
            datetime_series = datetime_series.str.to_datetime()

        seconds_since_epoch = (datetime_series.dt.timestamp() // 1_000_000).to_numpy()
        period_seconds = get_duration(rating_period)
        return ((seconds_since_epoch - seconds_since_epoch[0]) // period_seconds).astype(np.int32)

    def _process_time_steps(self):
        """Calculate time period boundaries."""
        self.unique_time_steps, time_indices = np.unique(self.time_steps, return_index=True)
        self.time_step_end_idxs = np.roll(time_indices, -1)
        self.time_step_end_idxs[-1] = len(self.time_steps)

    def __iter__(self):
        """Iterate through rating periods."""
        start_idx = 0
        for time_step, end_idx in zip(self.unique_time_steps, self.time_step_end_idxs):
            yield self.matchups[start_idx:end_idx], self.outcomes[start_idx:end_idx], time_step
            start_idx = end_idx

    def __getitem__(self, key):
        if isinstance(key, slice):
            return self._create_temporal_slice(key)
        raise ValueError('Only slice indexing supported')

    def _create_temporal_slice(self, key: slice):
        """Create sliced temporal dataset."""
        slice_data = {
            'time_steps': self.time_steps[key],
            'matchups': self.matchups[key],
            'outcomes': self.outcomes[key],
            'competitors': self.competitors,
        }
        dataset = self.init_from_arrays(**slice_data)
        dataset._process_time_steps()
        return dataset

    @classmethod
    def init_from_arrays(cls, time_steps: np.ndarray, matchups: np.ndarray, 
                        outcomes: np.ndarray, competitors: list):
        """Factory method with temporal support."""
        dataset = super().init_from_arrays(matchups, outcomes, competitors)
        dataset.time_steps = time_steps
        dataset._process_time_steps()
        return dataset


def split_pair_dataset(dataset, test_fraction=0.2):
    split_idx = math.ceil(len(dataset) * (1.0 - test_fraction))
    train_dataset = dataset[:split_idx]
    test_dataset = dataset[split_idx:]
    print(f'Split into train_dataset of length {len(train_dataset)} and test_dataset of length {len(test_dataset)}')
    return train_dataset, test_dataset


def generate_pair_data(
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
    strengths = np.exp(strengths)  # Ensure strengths are positive

    probs = np.zeros(shape=(num_matchups, 3))  # p(comp_1 win), p(draw), p(comp_2 win)
    probs[:, 0] = strengths[:, 0] / (strengths[:, 0] + (theta * strengths[:, 1]))
    probs[:, 2] = strengths[:, 1] / (strengths[:, 1] + (theta * strengths[:, 0]))
    probs[:, 1] = 1.0 - probs[:, 0] - probs[:, 2]

    outcomes = rng.multinomial(n=1, pvals=probs)
    outcomes = np.argmax(outcomes, axis=1) / 2.0  # Map 0->0, 1->0.5, 2->1.0

    df = pl.DataFrame({
        'timestamp': timestamps,
        'competitor_1': matchups[:, 0],
        'competitor_2': matchups[:, 1],
        'outcome': outcomes,
    })
    df = (
        df.with_columns([
            (pl.col('timestamp') * 1000).cast(pl.Datetime('ms')).alias('date'),
            pl.concat_str([pl.lit('competitor_'), pl.col('competitor_1').cast(pl.Utf8)]).alias('competitor_1'),
            pl.concat_str([pl.lit('competitor_'), pl.col('competitor_2').cast(pl.Utf8)]).alias('competitor_2')
        ])
        .drop('timestamp')
    )
    return df