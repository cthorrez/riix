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
