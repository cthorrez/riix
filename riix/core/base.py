"""base class for online rating systems"""
from abc import ABC
from typing import List
import numpy as np


class OnlineRatingSystem(ABC):
    """base class for online rating systems"""

    def predict(self, time_step: int, matchups: np.ndarray):
        raise NotImplementedError

    def fit(self, time_step, matchups: np.ndarray, outcomes: np.ndarray):
        raise NotImplementedError

    def print_topk(self, k: int, idx_to_competitor: List[str]):
        raise NotImplementedError
