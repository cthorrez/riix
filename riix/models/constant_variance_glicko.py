"""
Constant Variance Glicko
Martin Ingram 2021
https://www.researchgate.net/publication/348511584_How_to_extend_Elo_a_Bayesian_perspective
"""
import math
import numpy as np
from riix.core.base import OnlineRatingSystem
from riix.utils.math_utils import sigmoid, sigmoid_scalar


class ConstantVarianceGlicko(OnlineRatingSystem):
    """Glicko but all competitors have equal and unchanging variance"""

    rating_dim = 1

    def __init__(
        self,
        competitors: list,
        initial_rating: float = 1500.0,
        rd: float = 81.7,  # for tennis Ingram found this should be 81.7
        b: float = math.log(10.0) / 400.0,
        update_method: str = 'iterative',
        dtype=np.float64,
    ):
        """Initialize the rating system"""
        super().__init__(competitors)
        self.ratings = np.zeros(shape=self.num_competitors, dtype=dtype) + initial_rating
        self.rd = rd
        self.b = b
        self.b2 = b**2.0

        # a whole bunch of stuff is constant :)
        self.rd2_inv = 1.0 / (rd**2.0)
        self.g_rd = 1.0 / math.sqrt(1.0 + (3.0 * (self.b2) * (rd**2.0)) / (math.pi**2.0))
        self.b_g_rd = b * self.g_rd
        self.b2_g_rd2 = self.b_g_rd**2.0

        if update_method == 'batched':
            self.update = self.batched_update
        elif update_method == 'iterative':
            self.update = self.iterative_update

    def predict(self, matchups: np.ndarray, time_step: int = None, set_cache: bool = False):
        """
        Generates predictions for a series of matchups between competitors.
        """
        ratings = self.ratings[matchups]
        return sigmoid(self.b_g_rd * (ratings[:, 0] - ratings[:, 1]))

    def get_pre_match_ratings(self, matchups: np.ndarray, **kwargs):
        raise NotImplementedError

    def batched_update(self, matchups, outcomes, use_cache, **kwargs):
        """
        Apply a single update based on all results of the rating period.

        Parameters:
            matchups: Matchup information for the rating period.
            outcomes: Results of the matchups.
        """
        raise NotImplementedError

    def iterative_update(self, matchups, outcomes, **kwargs):
        """
        Treats the matchups in the rating period as sequential events.

        Parameters:
            matchups: Sequential matchups in the rating period.
            outcomes: Results of each matchup.
        """
        # somebody on twitter told me .self was slow so we are getting these once first
        rd2_inv = self.rd2_inv
        b_g_rd = self.b_g_rd
        b2_g_rd2 = self.b2_g_rd2
        for matchup, outcome in zip(matchups, outcomes):
            r_1, r_2 = self.ratings[matchup]
            prob = sigmoid_scalar(b_g_rd * (r_1 - r_2))
            d_squared_inv = b2_g_rd2 * prob * (1.0 - prob)
            k = b_g_rd / (rd2_inv + d_squared_inv)
            delta = k * (outcome - prob)
            self.ratings[matchup[0]] += delta
            self.ratings[matchup[1]] -= delta

    def print_leaderboard(self, num_places):
        raise NotImplementedError
