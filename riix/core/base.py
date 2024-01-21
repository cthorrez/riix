"""base class for online rating systems"""
from abc import ABC
import numpy as np
from riix.utils import MatchupDataset


class OnlineRatingSystem(ABC):
    """base class for online rating systems"""

    def predict(
        self,
        time_step: int,
        matchups: np.ndarray,
        set_cache: bool = False,
    ):
        raise NotImplementedError

    def fit(self, time_step: int, matchups: np.ndarray, outcomes: np.ndarray, use_cache: bool = False):
        raise NotImplementedError

    def rate_dataset(self, dataset: MatchupDataset, return_competitor_info: bool = False):
        raise NotImplementedError
