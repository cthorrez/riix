"""Glicko"""
import math
import numpy as np
from riix.core.base import OnlineRatingSystem
from riix.utils.math_utils import sigmoid


Q = math.log(10.0) / 400.0
Q2 = Q**2.0
Q2_3 = 3.0 * Q2
PI2 = math.pi**2.0


def g(rating_dev):
    """the g function"""
    return 1 / np.sqrt(1.0 + (Q2_3 * np.square(rating_dev)) / PI2)


def expected_score(rating_diff, g_rating_dev):
    """expected score for comp_1"""
    return 1.0 / (1.0 + np.power(10.0, -g_rating_dev * (rating_diff) / 400.0))


class Glicko(OnlineRatingSystem):
    """the og glicko rating system shoutout to Mark"""

    def __init__(
        self,
        num_competitors: int,
        initial_rating: float = 1500.0,
        initial_rating_dev: float = 63.2,
        c: float = 0.06,
        dtype=np.float64,
    ):
        self.num_competitors = num_competitors
        self.ratings = np.zeros(shape=num_competitors, dtype=dtype) + initial_rating
        self.rating_devs = np.zeros(shape=num_competitors, dtype=dtype) + initial_rating_dev
        self.cache = {'probs': None}

    def predict(self, time_step: int, matchups: np.ndarray, set_cache: bool = False):
        """generate predictions"""
        ratings_1 = self.ratings[matchups[:, 0]]
        ratings_2 = self.ratings[matchups[:, 1]]
        rating_diff = ratings_1 - ratings_2
        probs = sigmoid(self.alpha * (rating_diff))
        if set_cache:
            self.cache['probs'] = probs
        return probs

    def fit(
        self,
        time_step: int,
        matchups: np.ndarray,
        outcomes: np.ndarray,
        use_cache: bool = False,
        update_method: str = 'batched',
    ):
        if update_method == 'batched':
            self.batched_update(matchups, outcomes, use_cache)
        elif update_method == 'iterative':
            self.iterative_update(matchups, outcomes)

    def batched_update(self, matchups, outcomes, use_cache):
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

    def iterative_update(self, matchups, outcomes):
        """treat the matchups in the rating period as if they were sequential"""
        for idx in range(matchups.shape[0]):
            comp_1, comp_2 = matchups[idx]
            diff = self.ratings[comp_1] - self.ratings[comp_2]
            prob = sigmoid(self.alpha * diff)
            update = self.k * (outcomes[idx] - prob)
            self.ratings[comp_1] += update
            self.ratings[comp_2] -= update
