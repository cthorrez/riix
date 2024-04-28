"""Glicko"""
import math
import numpy as np
from riix.core.base import OnlineRatingSystem
from riix.utils.math_utils import sigmoid, sigmoid_scalar
from riix.utils.constants import PI2, Q, Q2, Q2_3


class Glicko(OnlineRatingSystem):
    """
    Implements the original Glicko rating system, designed by Mark Glickman.

    This rating system is an improvement over the Elo rating system, introducing the concept of rating
    deviation and volatility to better account for the uncertainty in a player's true strength.
    """

    rating_dim = 2

    def __init__(
        self,
        competitors: list,
        initial_rating: float = 1500.0,
        initial_rating_dev: float = 350.0,
        c: float = 63.2,
        dtype=np.float64,
        update_method='iterative',
        # update_method='batched',
        do_weird_prob=False,
    ):
        """
        Initializes the Glicko rating system with the given parameters.

        Parameters:
            competitors (list): A list of competitors to be rated within the system.
            initial_rating (float, optional): The initial Glicko rating for new competitors. Defaults to 1500.0.
            initial_rating_dev (float, optional): The initial rating deviation for new competitors. Defaults to 350.0.
            c (float, optional): Constant used to adjust the rate of change of the rating deviation. Defaults to 63.2.
            dtype (data-type, optional): The desired data-type for the ratings and deviations arrays. Defaults to np.float64.
            update_method (str, optional): Method used for updating ratings ('iterative' or another specified method). Defaults to 'iterative'.
            do_weird_prob (bool, optional): If set to True, applies an alternative probability calculation. Defaults to False.
        """
        super().__init__(competitors)
        self.initial_rating_dev = initial_rating_dev
        self.c2 = c**2.0
        self.ratings = np.zeros(shape=self.num_competitors, dtype=dtype) + initial_rating
        self.rating_devs = np.zeros(shape=self.num_competitors, dtype=dtype) + initial_rating_dev
        self.has_played = np.zeros(shape=self.num_competitors, dtype=np.bool_)
        self.prev_time_step = -1
        self.do_weird_prob = do_weird_prob

        if update_method == 'batched':
            self.update = self.batched_update
        elif update_method == 'iterative':
            self.update = self.iterative_update

    @staticmethod
    def g_vector(rating_dev):
        """
        Calculates the g function as part of the Glicko rating system.

        This function is used to scale the rating deviation, affecting the impact of a game outcome
        as a function of the opponent's rating volatility.

        Parameters:
            rating_dev (float): The rating deviation of an opponent.

        Returns:
            float: The calculated g function result, used to scale the expected score between players.
        """
        return 1.0 / np.sqrt(1.0 + (Q2_3 * np.square(rating_dev)) / PI2)

    @staticmethod
    def g_scalar(rating_dev):
        return 1.0 / math.sqrt(1.0 + (Q2_3 * (rating_dev) ** 2.0) / PI2)

    # TODO should Glicko probs incorporate the dev increase?
    def predict(self, matchups: np.ndarray, time_step: int = None, set_cache: bool = False):
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
            combined_dev = self.g_vector(np.sqrt(np.square(rating_devs_1) + np.square(rating_devs_2)))
            probs = sigmoid(Q * combined_dev * rating_diffs)
        return probs

    def get_pre_match_ratings(self, matchups: np.ndarray, **kwargs):
        means = self.ratings[matchups]
        devs = self.rating_devs[matchups]
        ratings = np.concatenate((means[..., None], devs[..., None]), axis=2).reshape(means.shape[0], -1)
        return ratings

    def increase_rating_dev(self, time_step, matchups):
        """called once per period to model the increase in variance over time"""
        active_in_period = np.unique(matchups)
        self.has_played[active_in_period] = True
        time_delta = time_step - self.prev_time_step
        self.rating_devs[self.has_played] = np.minimum(
            np.sqrt(np.square(self.rating_devs[self.has_played]) + (time_delta * self.c2)), self.initial_rating_dev
        )
        self.prev_time_step = time_step
        return active_in_period

    def batched_update(self, matchups, outcomes, time_step, use_cache=False, **kwargs):
        """apply one update based on all of the results of the rating period"""
        active_in_period = self.increase_rating_dev(time_step, matchups)
        masks = np.equal(matchups[:, :, None], active_in_period[None, :])  # N x 2 x active

        ratings = self.ratings[matchups]
        rating_diffs = ratings[:, 0] - ratings[:, 1]
        g_rating_devs = self.g_vector(self.rating_devs[matchups])
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

    def iterative_update(self, matchups, outcomes, time_step, **kwargs):
        """treat the matchups in the rating period as if they were sequential"""
        self.increase_rating_dev(time_step, matchups)
        for idx in range(matchups.shape[0]):
            comp_1, comp_2 = matchups[idx]
            rating_diff = self.ratings[comp_1] - self.ratings[comp_2]
            g_rating_devs = self.g_vector(self.rating_devs[matchups[idx]])
            g_rating_devs_2 = np.square(g_rating_devs)
            prob_1 = sigmoid_scalar(Q * g_rating_devs[1] * rating_diff)
            prob_2 = sigmoid_scalar(-Q * g_rating_devs[0] * rating_diff)
            d2_1 = 1.0 / (Q2 * prob_1 * (1.0 - prob_1) * g_rating_devs_2[1])
            d2_2 = 1.0 / (Q2 * prob_2 * (1.0 - prob_2) * g_rating_devs_2[0])
            r1_num = Q * g_rating_devs[1] * (outcomes[idx] - prob_1)
            r2_num = Q * g_rating_devs[0] * (1.0 - outcomes[idx] - prob_2)
            r1_denom = (1.0 / (self.rating_devs[comp_1] ** 2.0)) + (1.0 / d2_1)
            r2_denom = (1.0 / (self.rating_devs[comp_2] ** 2.0)) + (1.0 / d2_2)

            self.ratings[comp_1] += r1_num / r1_denom
            self.ratings[comp_2] += r2_num / r2_denom
            self.rating_devs[comp_1] = 1.0 / math.sqrt(r1_denom)
            self.rating_devs[comp_2] = 1.0 / math.sqrt(r2_denom)

    def print_leaderboard(self, num_places):
        sort_array = self.ratings - (3.0 * self.rating_devs)
        sorted_idxs = np.argsort(-sort_array)[:num_places]
        max_len = min(np.max([len(comp) for comp in self.competitors] + [10]), 25)
        print(f'{"competitor": <{max_len}}\t{"rating - (3*dev)"}\t')
        for p_idx in range(num_places):
            comp_idx = sorted_idxs[p_idx]
            print(f'{self.competitors[comp_idx]: <{max_len}}\t{sort_array[comp_idx]:.6f}')
