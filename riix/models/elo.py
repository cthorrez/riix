"""Elo"""
import math
import numpy as np
from riix.core.base import OnlineRatingSystem
from riix.utils.math_utils import sigmoid


class Elo(OnlineRatingSystem):
    """The Original Elo Rating system"""

    rating_dim = 1

    def __init__(
        self,
        num_competitors: int,
        initial_rating: float = 1500.0,
        k: float = 32.0,
        alpha: float = math.log(10.0) / 400.0,
        update_method: str = 'iterative',
        dtype=np.float64,
    ):
        self.num_competitors = num_competitors
        self.k = k
        self.alpha = alpha
        self.ratings = np.zeros(shape=num_competitors, dtype=dtype) + initial_rating
        self.cache = {'probs': None}
        if update_method == 'batched':
            self.update = self.batched_update
        elif update_method == 'iterative':
            self.update = self.iterative_update

    def predict(self, matchups: np.ndarray, time_step: int = None, set_cache: bool = False):
        """generate predictions"""
        ratings_1 = self.ratings[matchups[:, 0]]
        ratings_2 = self.ratings[matchups[:, 1]]
        probs = sigmoid(self.alpha * (ratings_1 - ratings_2))
        if set_cache:
            self.cache['probs'] = probs
        return probs

    # def fit(
    #     self,
    #     time_step: int,
    #     matchups: np.ndarray,
    #     outcomes: np.ndarray,
    #     use_cache: bool = False,
    # ):
    #     self.update(matchups, outcomes, use_cache=use_cache)

    def batched_update(self, matchups, outcomes, use_cache, return_ratings=False):
        """apply one update based on all of the results of the rating period"""
        active_in_period = np.unique(matchups)
        masks = np.equal(matchups[:, :, None], active_in_period[None, :])  # N x 2 x active
        if use_cache:
            probs = self.cache['probs']
        else:
            probs = self.predict(time_step=None, matchups=matchups, set_cache=False)
        per_match_diff = (outcomes - probs)[:, None]
        per_match_diff = np.hstack([per_match_diff, -per_match_diff])
        per_competitor_diff = (per_match_diff[:, :, None] * masks).sum(axis=(0, 1))
        self.ratings[active_in_period] += self.k * per_competitor_diff

    def iterative_update(self, matchups, outcomes, return_ratings=False, **kwargs):
        """treat the matchups in the rating period as if they were sequential"""
        ratings = self.ratings[matchups]
        for idx in range(matchups.shape[0]):
            comp_1, comp_2 = matchups[idx]
            diff = self.ratings[comp_1] - self.ratings[comp_2]
            prob = sigmoid(self.alpha * diff)
            update = self.k * (outcomes[idx] - prob)
            self.ratings[comp_1] += update
            self.ratings[comp_2] -= update
        if return_ratings:
            return ratings

    def print_top_k(self, k, competitor_names):
        sorted_idxs = np.argsort(-self.ratings)[:k]
        max_len = np.max([len(name) for name in competitor_names] + [10])
        print(f'{"competitor": <{max_len}}\t{"rating"}')
        for k_idx in range(k):
            comp_idx = sorted_idxs[k_idx]
            print(f'{competitor_names[comp_idx]: <{max_len}}\t{self.ratings[comp_idx]:.6f}')
