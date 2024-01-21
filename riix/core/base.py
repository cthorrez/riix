"""base class for online rating systems"""
from abc import ABC
from typing import Optional, Dict, NamedTuple
import numpy as np
from riix.utils import MatchupDataset


class RatingOutputBatch(NamedTuple):
    """class to hold the outputs of a rating system for one batch of data"""

    probs: np.ndarray
    competitor_data: Optional[Dict[str, np.ndarray]]  # this can be ratings, rating deviations etc


class OnlineRatingSystem(ABC):
    """base class for online rating systems"""

    def predict(
        self,
        matchups: np.ndarray,
        time_step: int = None,
        set_cache: bool = False,
    ):
        raise NotImplementedError

    def fit(self, time_step: int, matchups: np.ndarray, outcomes: np.ndarray, use_cache: bool = False):
        raise NotImplementedError

    def rate_batch(
        self,
        batch,
        return_competitor_info: bool = False,
        cache: bool = False,
    ) -> RatingOutputBatch:
        raise NotImplementedError

    def rate_dataset(
        self,
        dataset: MatchupDataset,
        return_competitor_info: bool = False,
        cache: bool = False,
    ):
        """evaluate a rating system on a dataset"""
        all_probs = np.zeros(len(dataset))
        # all_ratings = np.zeros(size=(len(dataset),2))
        probs_idx = 0
        for input_batch in dataset:
            output_batch = self.rate_batch(input_batch)
            batch_probs = output_batch.probs
            all_probs[probs_idx : probs_idx + batch_probs.shape[0]] = batch_probs
            probs_idx += batch_probs.shape[0]
        return all_probs
