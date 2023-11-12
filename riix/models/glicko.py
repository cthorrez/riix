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
    return 1.0 / np.sqrt(1.0 + (Q2_3 * np.square(rating_dev)) / PI2)


class Glicko(OnlineRatingSystem):
    """the og glicko rating system shoutout to Mark"""

    def __init__(
        self,
        num_competitors: int,
        initial_rating: float = 1500.0,
        initial_rating_dev: float = 350.0,
        c: float = 63.2,
        dtype=np.float64,
        do_weird_prob=False,
    ):
        self.num_competitors = num_competitors
        self.initial_rating_dev = initial_rating_dev
        self.c2 = c**2.0
        self.ratings = np.zeros(shape=num_competitors, dtype=dtype) + initial_rating
        self.rating_devs = np.zeros(shape=num_competitors, dtype=dtype) + initial_rating_dev
        self.has_played = np.zeros(shape=num_competitors, dtype=np.bool_)
        self.do_weird_prob = do_weird_prob

    def predict(self, time_step: int, matchups: np.ndarray, set_cache: bool = False):
        """generate predictions"""
        ratings_1 = self.ratings[matchups[:, 0]]
        ratings_2 = self.ratings[matchups[:, 1]]
        rating_diffs = ratings_1 - ratings_2
        if self.do_weird_prob:
            # not sure why they do it this way but it seems to work better than the "real" way
            # https://github.com/McLeopold/PythonSkills/blob/95559262fbeaabc39cc5d698b93a6e43dc9b5e64/skills/glicko.py#L181
            probs = 1.0 / (1.0 + np.power(10, -rating_diffs / (2.0 * self.initial_rating_dev)))
        else:
            rating_devs_1 = self.rating_devs[matchups[:, 0]]
            rating_devs_2 = self.rating_devs[matchups[:, 1]]
            combined_dev = g(np.sqrt(np.square(rating_devs_1) + np.square(rating_devs_2)))
            probs = sigmoid(Q * combined_dev * rating_diffs)
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

    def increase_rating_dev(self, matchups):
        """called once per period to model the increase in variance over time"""
        active_in_period = np.unique(matchups)
        self.has_played[active_in_period] = True
        self.rating_devs[self.has_played] = np.minimum(
            np.sqrt(np.square(self.rating_devs[self.has_played]) + self.c2), self.initial_rating_dev
        )
        return active_in_period

    def batched_update(self, matchups, outcomes, use_cache=False):
        """apply one update based on all of the results of the rating period"""
        active_in_period = self.increase_rating_dev(matchups)
        masks = np.equal(matchups[:, :, None], active_in_period[None, :])  # N x 2 x active

        ratings = self.ratings[matchups]
        rating_diffs = ratings[:, 0] - ratings[:, 1]
        g_rating_devs = g(self.rating_devs[matchups])
        probs_1 = sigmoid(Q * g_rating_devs[:, 1] * rating_diffs)
        probs_2 = sigmoid(-1.0 * (Q * g_rating_devs[:, 0] * rating_diffs))

        tmp = np.stack([probs_1 * (1.0 - probs_1), probs_2 * (1.0 - probs_2)]).T * np.square(g_rating_devs)[:, [1, 0]]
        d2 = 1.0 / ((tmp[:, :, None] * masks).sum(axis=(0, 1)) * Q2)

        outcomes = np.hstack([outcomes[:, None], 1.0 - outcomes[:, None]])
        probs = np.hstack([probs_1[:, None], probs_2[:, None]])

        r_num = Q * ((g_rating_devs[:, [1, 0]] * (outcomes - probs))[:, :, None] * masks).sum(axis=(0, 1))
        r_denom = (1.0 / np.square(self.rating_devs[active_in_period])) + (1.0 / d2)

        self.ratings[active_in_period] += r_num / r_denom
        self.rating_devs[active_in_period] = np.sqrt(1.0 / r_denom)

    def iterative_update(self, matchups, outcomes):
        """treat the matchups in the rating period as if they were sequential"""
        self.increase_rating_dev(matchups)
        for idx in range(matchups.shape[0]):
            comp_1, comp_2 = matchups[idx]
            rating_diff = self.ratings[comp_1] - self.ratings[comp_2]
            g_rating_devs = g(self.rating_devs[matchups[idx]])
            g_rating_devs_2 = np.square(g_rating_devs)
            prob_1 = sigmoid(Q * g_rating_devs[1] * rating_diff)
            prob_2 = sigmoid(-Q * g_rating_devs[0] * rating_diff)
            d2_1 = 1.0 / (Q2 * prob_1 * (1.0 - prob_1) * g_rating_devs_2[1])
            d2_2 = 1.0 / (Q2 * prob_2 * (1.0 - prob_2) * g_rating_devs_2[0])
            r1_num = Q * g_rating_devs[1] * (outcomes[idx] - prob_1)
            r2_num = Q * g_rating_devs[0] * (1.0 - outcomes[idx] - prob_2)
            r1_denom = (1.0 / (self.rating_devs[comp_1] ** 2.0)) + (1.0 / d2_1)
            r2_denom = (1.0 / (self.rating_devs[comp_2] ** 2.0)) + (1.0 / d2_2)
            self.ratings[comp_1] += r1_num / r1_denom
            self.ratings[comp_2] += r2_num / r2_denom
            self.rating_devs[comp_1] = math.sqrt(1.0 / r1_denom)
            self.rating_devs[comp_2] = math.sqrt(1.0 / r2_denom)
