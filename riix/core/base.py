"""base class for online rating systems"""
from abc import ABC
from typing import Optional
import numpy as np
from riix.utils import MatchupDataset


class OnlineRatingSystem(ABC):
    """
    Base class for online rating systems. This class provides a framework for implementing
    various online rating systems, such as Elo, Glicko, or TrueSkill. It defines the basic
    structure and common methods that all such systems might share.

    Attributes:
        rating_dim (int): Dimension of competitor ratings. This could be 1 for systems like Elo,
                          where each competitor has a single rating value, or more for systems
                          like TrueSkill that use multiple values (e.g., mean and standard deviation).
        competitors (list): A list of competitors within the rating system.
        num_competitors (int): The number of competitors in the system.
    """

    rating_dim: int

    def __init__(self, competitors):
        """
        Initializes a new instance of an online rating system with a list of competitors.

        Parameters:
            competitors (list): A list of competitors to be included in the rating system. Each
                                competitor should have a structure or identifier compatible with
                                the specific rating system's requirements.
        """
        self.competitors = competitors
        self.num_competitors = len(competitors)

    def print_leaderboard(self, num_places=None):
        """
        Prints the leaderboard of the rating system.

        Parameters:
            num_places int: The number of top places to display on the leaderboard.
        """
        pass  # Implementation should be provided by subclasses.

    def predict(
        self,
        matchups: np.ndarray,
        time_step: int = None,
        set_cache: bool = False,
    ):
        raise NotImplementedError

    def update(self, matchups: np.ndarray, outcomes: np.ndarray, time_step: Optional[int], use_cache: bool = False):
        """
        Updates player ratings based on new matchup results.
        This general update function processes a set of matchups and their outcomes to adjust player ratings within the model.

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

    def iterative_update(self, matchups: np.ndarray, outcomes: np.ndarray, time_step: int, use_cache=False, **kwargs):
        """
        Updates player ratings iteratively for each matchup and outcome pair within a given time step treating them as if they were sequential
        This can be slower than batch processing but potentially more accurate by avoiding certain numerical issues

        Parameters:
            matchups (np.ndarray): Array of matchups, where each matchup is represented by a pair of player indices
            outcomes (np.ndarray): Array of outcomes corresponding to each matchup represented as win (1), loss (0), or draw (0.5).
            time_step (int): The current time step or period of the rating update, used to adjust ratings over time.
            use_cache (bool, optional): Whether to use values cached during a prior call to predict() to speed up calculations. Defaults to False.
        """
        raise NotImplementedError

    def get_pre_match_ratings(self, matchups: np.ndarray, time_step: Optional[int]) -> np.ndarray:
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

    def fit_batch(
        self,
        matchups: np.ndarray,
        outcomes: np.ndarray,
        time_step: int = None,
        return_pre_match_probs: bool = False,
        return_pre_match_ratings: bool = False,
        cache: bool = False,
    ) -> dict:
        if return_pre_match_probs:
            pre_match_probs = self.predict(matchups=matchups, time_step=time_step, set_cache=cache)
        if return_pre_match_ratings:
            pre_match_ratings = self.get_pre_match_ratings(matchups, time_step=time_step)
        self.update(matchups, outcomes, time_step=time_step, use_cache=cache)
        if return_pre_match_probs and return_pre_match_ratings:
            return pre_match_probs, pre_match_ratings
        elif return_pre_match_probs:
            return pre_match_probs
        elif return_pre_match_ratings:
            return_pre_match_ratings

    def fit_dataset(
        self,
        dataset: MatchupDataset,
        return_pre_match_probs: bool = False,
        return_pre_match_ratings: bool = False,
        cache: bool = False,
    ):
        """evaluate a rating system on a dataset"""
        n_matchups = len(dataset)
        if return_pre_match_probs:
            pre_match_probs = np.empty(shape=(n_matchups))
        if return_pre_match_ratings:
            pre_match_ratings = np.empty(shape=(n_matchups, 2 * self.rating_dim))

        idx = 0
        for matchups, outcomes, time_step in dataset:
            batch_outputs = self.fit_batch(
                matchups=matchups,
                outcomes=outcomes,
                time_step=time_step,
                return_pre_match_probs=return_pre_match_probs,
                return_pre_match_ratings=return_pre_match_ratings,
            )
            if (batch_outputs is not None) and (not isinstance(batch_outputs, tuple)):
                batch_outputs = (batch_outputs,)
            if return_pre_match_probs:
                batch_probs = batch_outputs[0]
                pre_match_probs[idx : idx + batch_probs.shape[0]] = batch_probs
            if return_pre_match_ratings:
                batch_pre_match_ratings = batch_outputs[-1]
                pre_match_ratings[idx : idx + batch_pre_match_ratings.shape[0]] = batch_pre_match_ratings
            if batch_outputs:
                idx += batch_outputs[0].shape[0]

        if return_pre_match_probs and return_pre_match_ratings:
            return pre_match_probs, pre_match_ratings
        elif return_pre_match_probs:
            return pre_match_probs
        elif return_pre_match_ratings:
            return_pre_match_ratings
