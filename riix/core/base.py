"""base class for online rating systems"""
from abc import ABC
import numpy as np
from riix.utils import MatchupDataset


class OnlineRatingSystem(ABC):
    """base class for online rating systems"""

    rating_dim: int  # dimension of competitor ratings, eg 1 for Elo rating, 2 for TrueSkill mu and sigma

    def predict(
        self,
        matchups: np.ndarray,
        time_step: int = None,
        set_cache: bool = False,
    ):
        raise NotImplementedError

    def fit(self, time_step: int, matchups: np.ndarray, outcomes: np.ndarray, use_cache: bool = False):
        raise NotImplementedError

    def update(self, matchups: np.ndarray, outcomes: np.ndarray, time_step: int, use_cache: bool = False):
        raise NotImplementedError

    def rate_batch(
        self,
        matchups: np.ndarray,
        outcomes: np.ndarray,
        time_step: int = None,
        return_ratings: bool = False,
        cache: bool = False,
    ):
        probs = self.predict(matchups=matchups, set_cache=cache)
        ratings = self.update(matchups, outcomes, time_step=time_step, use_cache=cache, return_ratings=return_ratings)
        if not return_ratings:
            return probs
        return probs, ratings

    def rate_dataset(
        self,
        dataset: MatchupDataset,
        return_ratings: bool = False,
        cache: bool = False,
    ):
        """evaluate a rating system on a dataset"""
        all_probs = np.zeros(len(dataset))
        if return_ratings:
            all_ratings = np.zeros(shape=(len(dataset), self.rating_dim * 2))
        idx = 0
        for matchups, outcomes, time_step in dataset:
            if not return_ratings:
                probs = self.rate_batch(matchups=matchups, outcomes=outcomes, time_step=time_step)
            else:
                probs, ratings = self.rate_batch(
                    matchups=matchups, outcomes=outcomes, time_step=time_step, return_ratings=True
                )
                all_ratings[idx : idx + ratings.shape[0]] = ratings
            all_probs[idx : idx + probs.shape[0]] = probs
            idx += probs.shape[0]
        if not return_ratings:
            return all_probs
        else:
            return all_probs, all_ratings
