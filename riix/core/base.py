"""base class for online rating systems"""
from abc import ABC
import numpy as np


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
