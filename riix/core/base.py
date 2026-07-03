"""Base classes for rating systems."""

from abc import ABC
from typing import NamedTuple, Optional
import numpy as np
from riix.utils import PairDataset, TimedPairDataset


class FitResult(NamedTuple):
    """
    Result of fitting a rating system on a batch or dataset.

    Fields are None unless the corresponding `return_pre_match_*` flag was passed,
    so `fit_dataset(...).probs` and `probs, ratings = fit_dataset(...)` both work
    with a single, consistent return type.
    """

    probs: Optional[np.ndarray]
    ratings: Optional[np.ndarray]


class RatingSystem(ABC):
    """
    Base class shared by all rating systems, online or static.

    Attributes:
        rating_dim (int): Dimension of competitor ratings. 1 for systems like Elo where each
                          competitor has a single rating value, more for systems like TrueSkill
                          that use multiple values (e.g., mean and standard deviation).
        competitors (list): A list of competitors within the rating system.
        num_competitors (int): The number of competitors in the system.
        leaderboard_header (str): Column header printed above the scores by print_leaderboard.
    """

    rating_dim: int
    leaderboard_header: str = 'rating'

    def __init__(self, competitors):
        """
        Initializes a new instance of a rating system with a list of competitors.

        Parameters:
            competitors (list): A list of competitors to be included in the rating system. Each
                                competitor should have a structure or identifier compatible with
                                the specific rating system's requirements.
        """
        self.competitors = competitors
        self.num_competitors = len(competitors)

    def predict(
        self,
        matchups: np.ndarray,
        time_step: int = None,
        set_cache: bool = False,
    ):
        raise NotImplementedError

    def get_pre_match_ratings(self, matchups: np.ndarray, time_step: Optional[int] = None) -> np.ndarray:
        """
        Returns the ratings for competitors at the timestep of the matchups
        Useful when using pre-match ratings as features in downstream ML pipelines

        Parameters:
            matchups (np.ndarray of shape (n,2)): competitor indices
            time_step (optional int)

        Returns:
            np.ndarray of shape (n,2): ratings for specified competitors
        """
        raise NotImplementedError

    def fit_dataset(
        self,
        dataset: PairDataset,
        return_pre_match_probs: bool = False,
        return_pre_match_ratings: bool = False,
        cache: bool = False,
    ) -> FitResult:
        """Fit the rating system on a dataset."""
        raise NotImplementedError

    def leaderboard_ratings(self) -> np.ndarray:
        """
        Returns the array of shape (num_competitors,) of scores used to rank competitors
        on the leaderboard. Override per rating system (e.g. `mus - 3 * sqrt(sigma2s)`).
        """
        raise NotImplementedError

    def print_leaderboard(self, num_places: int = None):
        """
        Prints the leaderboard of the rating system, ranked by leaderboard_ratings().

        Parameters:
            num_places (int, optional): The number of top places to display.
                                        Defaults to all competitors.
        """
        scores = np.asarray(self.leaderboard_ratings())
        if num_places is None:
            num_places = self.num_competitors
        sorted_idxs = np.argsort(-scores)[:num_places]
        max_len = min(max([len(str(comp)) for comp in self.competitors] + [10]), 25)
        print(f'{"competitor": <{max_len}}\t{self.leaderboard_header}')
        for comp_idx in sorted_idxs:
            print(f'{str(self.competitors[comp_idx]): <{max_len}}\t{scores[comp_idx]:.6f}')


class OnlineRatingSystem(RatingSystem):
    """
    Base class for online (dynamic) rating systems such as Elo, Glicko, or TrueSkill,
    whose ratings evolve as matchups are processed period by period.
    """

    def __init__(self, competitors, update_method: str = 'online'):
        """
        Initializes a new instance of an online rating system.

        Parameters:
            competitors (list): A list of competitors to be included in the rating system.
            update_method (str, optional): 'online' processes the matchups within a rating period
                                           sequentially, 'batched' applies one simultaneous update
                                           per rating period. Defaults to 'online'.
        """
        super().__init__(competitors)
        if update_method == 'batched':
            self.update = self.batched_update
        elif update_method == 'online':
            self.update = self.online_update
        else:
            raise ValueError(f'update_method must be one of online or batched, got {update_method}')

    def update(self, matchups: np.ndarray, outcomes: np.ndarray, time_step: Optional[int], use_cache: bool = False):
        """
        Updates player ratings based on new matchup results.
        This general update function processes a set of matchups and their outcomes to adjust player ratings within the model.
        It is bound to either batched_update or online_update by the constructor's update_method argument.

        Parameters:
            matchups (np.ndarray): Array of matchups, where each matchup is represented by a pair of player indices
            outcomes (np.ndarray): Array of outcomes corresponding to each matchup represented as win (1), loss (0), or draw (0.5).
            time_step (int): The current time step or period of the rating update, used to adjust ratings over time.
            use_cache (bool, optional): Whether to use values cached during a prior call to predict() to speed up calculations. Defaults to False.
        """
        raise NotImplementedError

    def batched_update(self, matchups: np.ndarray, outcomes: np.ndarray, time_step: int, use_cache=False, **kwargs):
        """
        Performs a batched update of player ratings based on a series of matchups and their outcomes within a single time step.
        This method processes all matchups as occuring simultaneously, leveraging vectorized operations for efficiency

        Parameters:
            matchups (np.ndarray): Array of matchups, where each matchup is represented by a pair of player indices
            outcomes (np.ndarray): Array of outcomes corresponding to each matchup represented as win (1), loss (0), or draw (0.5).
            time_step (int): The current time step or period of the rating update, used to adjust ratings over time.
            use_cache (bool, optional): Whether to use values cached during a prior call to predict() to speed up calculations. Defaults to False.
        """
        raise NotImplementedError

    def online_update(self, matchups: np.ndarray, outcomes: np.ndarray, time_step: int, use_cache=False, **kwargs):
        """
        Updates player ratings onlinely for each matchup and outcome pair within a given time step treating them as if they were sequential
        This can be slower than batch processing but potentially more accurate by avoiding certain numerical issues

        Parameters:
            matchups (np.ndarray): Array of matchups, where each matchup is represented by a pair of player indices
            outcomes (np.ndarray): Array of outcomes corresponding to each matchup represented as win (1), loss (0), or draw (0.5).
            time_step (int): The current time step or period of the rating update, used to adjust ratings over time.
            use_cache (bool, optional): Whether to use values cached during a prior call to predict() to speed up calculations. Defaults to False.
        """
        raise NotImplementedError

    def fit_batch(
        self,
        matchups: np.ndarray,
        outcomes: np.ndarray,
        time_step: int = None,
        return_pre_match_probs: bool = False,
        return_pre_match_ratings: bool = False,
        cache: bool = False,
    ) -> FitResult:
        """Predict and update on the matchups of a single rating period."""
        pre_match_probs = None
        pre_match_ratings = None
        if return_pre_match_probs:
            pre_match_probs = self.predict(matchups=matchups, time_step=time_step, set_cache=cache)
        if return_pre_match_ratings:
            pre_match_ratings = self.get_pre_match_ratings(matchups, time_step=time_step)
        self.update(matchups, outcomes, time_step=time_step, use_cache=cache)
        return FitResult(probs=pre_match_probs, ratings=pre_match_ratings)

    def fit_dataset(
        self,
        dataset: TimedPairDataset,
        return_pre_match_probs: bool = False,
        return_pre_match_ratings: bool = False,
        cache: bool = False,
    ) -> FitResult:
        """Fit the rating system on a dataset, iterating over its rating periods."""
        n_matchups = len(dataset)
        pre_match_probs = np.empty(shape=(n_matchups,)) if return_pre_match_probs else None
        pre_match_ratings = np.empty(shape=(n_matchups, 2 * self.rating_dim)) if return_pre_match_ratings else None

        idx = 0
        for matchups, outcomes, time_step in dataset:
            batch_result = self.fit_batch(
                matchups=matchups,
                outcomes=outcomes,
                time_step=time_step,
                return_pre_match_probs=return_pre_match_probs,
                return_pre_match_ratings=return_pre_match_ratings,
                cache=cache,
            )
            batch_size = matchups.shape[0]
            if return_pre_match_probs:
                pre_match_probs[idx : idx + batch_size] = batch_result.probs
            if return_pre_match_ratings:
                pre_match_ratings[idx : idx + batch_size] = batch_result.ratings
            idx += batch_size

        return FitResult(probs=pre_match_probs, ratings=pre_match_ratings)


class StaticRatingSystem(RatingSystem):
    """
    Base class for static (non-dynamic) rating systems, which fit a single fixed set of
    ratings to the entire dataset at once (e.g. batch MLE Bradley-Terry, Massey) instead
    of evolving ratings match by match.
    """

    def fit(self, matchups: np.ndarray, outcomes: np.ndarray, time_steps: Optional[np.ndarray] = None):
        """
        Fit the ratings on all matchups at once.

        Parameters:
            matchups (np.ndarray): (n, 2) array of competitor indices.
            outcomes (np.ndarray): Array of outcomes represented as win (1), loss (0), or draw (0.5).
            time_steps (np.ndarray, optional): Time step of each matchup, for systems that weight by recency.
        """
        raise NotImplementedError

    def fit_dataset(
        self,
        dataset: PairDataset,
        return_pre_match_probs: bool = False,
        return_pre_match_ratings: bool = False,
        cache: bool = False,
    ) -> FitResult:
        """
        Fit on the whole dataset at once, then predict its matchups in-sample.

        Note that unlike an online system, the returned probs and ratings come from the
        ratings fit on ALL matchups, they are in-sample rather than truly "pre-match".
        The signature matches OnlineRatingSystem.fit_dataset so that riix.eval works
        with both kinds of systems.
        """
        self.fit(dataset.matchups, dataset.outcomes, time_steps=getattr(dataset, 'time_steps', None))
        probs = self.predict(dataset.matchups) if return_pre_match_probs else None
        ratings = self.get_pre_match_ratings(dataset.matchups) if return_pre_match_ratings else None
        return FitResult(probs=probs, ratings=ratings)
