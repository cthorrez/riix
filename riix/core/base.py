"""base class for online rating systems"""
from abc import ABC
from typing import Optional
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

    def update(self, matchups: np.ndarray, outcomes: np.ndarray, time_step: int, use_cache: bool = False):
        raise NotImplementedError

    def get_pre_match_ratings(self, matchups: np.ndarray, time_step: Optional[int]) -> np.ndarray:
        raise NotImplementedError

    def rate_batch(
        self,
        matchups: np.ndarray,
        outcomes: np.ndarray,
        time_step: int = None,
        return_pre_match_ratings: bool = False,
        cache: bool = False,
    ):
        if return_pre_match_ratings:
            pre_match_ratings = self.get_pre_match_ratings(matchups, time_step=time_step)
        probs = self.predict(matchups=matchups, time_step=time_step, set_cache=cache)
        self.update(matchups, outcomes, time_step=time_step, use_cache=cache)
        if not return_pre_match_ratings:
            return probs
        return probs, pre_match_ratings

    def rate_dataset(
        self,
        dataset: MatchupDataset,
        return_pre_match_ratings: bool = False,
        cache: bool = False,
    ):
        """evaluate a rating system on a dataset"""
        all_probs = np.zeros(len(dataset))
        if return_pre_match_ratings:
            all_ratings = np.zeros(shape=(len(dataset), self.rating_dim * 2))
        idx = 0
        for matchups, outcomes, time_step in dataset:
            if not return_pre_match_ratings:
                probs = self.rate_batch(matchups=matchups, outcomes=outcomes, time_step=time_step)
            else:
                probs, ratings = self.rate_batch(
                    matchups=matchups, outcomes=outcomes, time_step=time_step, return_pre_match_ratings=True
                )
                all_ratings[idx : idx + ratings.shape[0]] = ratings
            all_probs[idx : idx + probs.shape[0]] = probs
            idx += probs.shape[0]
        if not return_pre_match_ratings:
            return all_probs
        else:
            return all_probs, all_ratings
