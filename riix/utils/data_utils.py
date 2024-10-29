"""classes and functions for working with rating data"""

import math
import time
from typing import List
import numpy as np
import polars as pl
from riix.utils.date_utils import get_duration

class MatchupDataset:
    """class for loading and iterating over paired comparison data"""

    def __init__(
        self,
        df: pl.DataFrame,
        competitor_cols: List[str],
        outcome_col: str,
        datetime_col: str = None,
        time_step_col: str = None,
        rating_period: str = '1W',
        verbose: bool = True,
    ):
        if len(competitor_cols) != 2:
            raise ValueError('must specify exactly 2 competitor columns')
        if (bool(datetime_col) + bool(time_step_col)) != 1:
            raise ValueError('must specify only one of time_step_col, datetime_col, timestamp_col')
        if time_step_col:
            self.time_steps = df[time_step_col]
        else:
            if df[datetime_col].dtype == pl.Datetime("ms"):
                datetime = df[datetime_col]
            elif df.schema[datetime_col] == pl.Date:
                datetime = df[datetime_col].cast(pl.Datetime)
            elif df.schema[datetime_col] == pl.Utf8:
                datetime = df.with_columns(
                    pl.when(pl.col(datetime_col).str.contains(r'^\d{4}-\d{2}-\d{2}$'))  # Match 'yyyy-mm-dd'
                    .then(pl.col(datetime_col).str.to_date('%Y-%m-%d'))
                    .otherwise(pl.col(datetime_col).str.strptime(pl.Datetime, '%Y-%m-%dT%H:%M:%S%.f', strict=False))
                )[datetime_col]
            else:
                raise ValueError('datetime_col must be one of Date, Datetime, or Utf8')
            seconds_since_epoch = (datetime.dt.timestamp() // 1_000_000).to_numpy()
            rating_period_duration_in_seconds = get_duration(rating_period)
            first_time = seconds_since_epoch[0]
            seconds_since_first_time = seconds_since_epoch - first_time
            self.time_steps = seconds_since_first_time // rating_period_duration_in_seconds

        self.time_steps = self.time_steps.astype(np.int32)
        self.process_time_steps()

        # Create a single competitors reference dataframe
        competitors_df = pl.DataFrame(
            {'competitor': pl.concat([
                df[competitor_cols[0]].cast(pl.Utf8),
                df[competitor_cols[1]].cast(pl.Utf8)
            ]).unique().sort()
        }).lazy().select(
            pl.all(),
            pl.int_range(pl.len(), dtype=pl.Int32).alias('index')
        )
        self.competitors = sorted(competitors_df.collect()['competitor'].to_list())
        self.num_competitors = len(self.competitors)
        self.competitor_to_idx = dict(zip(self.competitors, range(self.num_competitors)))
        matchups_df = (df.lazy()
            .select([
                pl.col(competitor_cols[0]).cast(pl.Utf8).alias('comp1'),
                pl.col(competitor_cols[1]).cast(pl.Utf8).alias('comp2')
            ])
            .join(
                competitors_df,
                left_on='comp1',
                right_on='competitor'
            ).rename({'index': 'index1'})
            .join(
                competitors_df,
                left_on='comp2',
                right_on='competitor'
            ).rename({'index': 'index2'})
            .select(['index1', 'index2'])
        )
        self.matchups = np.ascontiguousarray(matchups_df.collect().to_numpy())



        self.outcomes = df[outcome_col].to_numpy()

        if verbose:
            print('loaded dataset with:')
            print(f'{self.matchups.shape[0]} matchups')
            print(f'{len(self.competitors)} unique competitors')
            print(f'{self.unique_time_steps.max() + 1} rating periods of length {rating_period}')

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
