"""
Glicko 2
paper: http://www.glicko.net/research/dpcmsv.pdf
example: http://www.glicko.net/glicko/glicko2.pdf

"""
import math
import numpy as np
from riix.core.base import OnlineRatingSystem
from riix.utils.math_utils import sigmoid
from riix.utils.constants import THREE_OVER_PI_SQUARED


class Glicko2(OnlineRatingSystem):
    """
    Implements the Glicko 2 rating system, designed by Mark Glickman.
    """

    rating_dim = 2

    def __init__(
        self,
        competitors: list,
        initial_rating: float = 1500.0,
        initial_rating_dev: float = 350.0,
        initial_sigma: float = 0.06,
        tau: float = 0.2,
        dtype=np.float64,
        update_method='iterative',
    ):
        """Initializes the Glicko rating system with the given parameters."""
        super().__init__(competitors)
        self.mus = np.zeros(shape=self.num_competitors, dtype=dtype) + (initial_rating - 1500.00) / 173.7178
        self.phis = np.zeros(shape=self.num_competitors, dtype=dtype) + (initial_rating_dev / 173.7178)
        self.sigmas = np.zeros(shape=self.num_competitors, dtype=dtype) + initial_sigma
        self.has_played = np.zeros(shape=self.num_competitors, dtype=np.bool_)

        if update_method == 'batched':
            self.update = self.batched_update
        elif update_method == 'iterative':
            self.update = self.iterative_update

    @staticmethod
    def g_scalar(phi):
        """this is DIFFERENT from g in regular Glicko"""
        return 1.0 / math.sqrt(1.0 + (THREE_OVER_PI_SQUARED * (phi**2.0)))

    # TODO should Glicko probs incorporate the dev increase?
    def predict(self, matchups: np.ndarray, time_step: int = None, set_cache: bool = False):
        """generate predictions"""
        ratings_1 = self.ratings[matchups[:, 0]]
        ratings_2 = self.ratings[matchups[:, 1]]
        rating_diffs = ratings_1 - ratings_2
        rating_devs_1 = self.rating_devs[matchups[:, 0]]
        rating_devs_2 = self.rating_devs[matchups[:, 1]]
        combined_dev = self.g(np.sqrt(np.square(rating_devs_1) + np.square(rating_devs_2)))
        probs = sigmoid(combined_dev * rating_diffs)
        return probs

    def get_pre_match_ratings(self, matchups: np.ndarray, **kwargs):
        means = self.ratings[matchups]
        devs = self.rating_devs[matchups]
        ratings = np.concatenate((means[..., None], devs[..., None]), axis=2).reshape(means.shape[0], -1)
        return ratings

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
        # TODO

    def iterative_update(self, matchups, outcomes, **kwargs):
        """treat the matchups in the rating period as if they were sequential"""
        self.increase_rating_dev(matchups)
        for idx in range(matchups.shape[0]):
            pass
            # comp_1, comp_2 = matchups[idx]
            # rating_diff = self.ratings[comp_1] - self.ratings[comp_2]

    def print_leaderboard(self, num_places):
        sort_array = self.ratings - (3.0 * self.rating_devs)
        sorted_idxs = np.argsort(-sort_array)[:num_places]
        max_len = min(np.max([len(comp) for comp in self.competitors] + [10]), 25)
        print(f'{"competitor": <{max_len}}\t{"rating - (3*dev)"}\t')
        for p_idx in range(num_places):
            comp_idx = sorted_idxs[p_idx]
            print(f'{self.competitors[comp_idx]: <{max_len}}\t{sort_array[comp_idx]:.6f}')
