"""
Bayesian generalization of Elo by Martin Ingram
https://www.researchgate.net/publication/348511584_How_to_extend_Elo_a_Bayesian_perspective
equations 17-19 are relevant for the case without adding surface, and best-of effects
"""
import math
import numpy as np
from riix.core.base import OnlineRatingSystem
from riix.utils.math_utils import sigmoid, sigmoid_scalar


class GenElo(OnlineRatingSystem):
    """Bayesian Generalized Elo from Ingram 2021"""

    rating_dim = 1

    def __init__(
        self,
        competitors: list,
        initial_mu: float = 0.0,
        b: float = math.log(10.0) / 400.0,  # the temperature of the softmax
        sigma: float = 60.0,
        use_approx: bool = True,
        update_method: str = 'iterative',
        dtype=np.float64,
    ):
        """Initialize the rating system"""
        super().__init__(competitors)
        self.mus = np.zeros(shape=self.num_competitors, dtype=dtype) + initial_mu
        self.b = b
        self.b_over_2 = b / 2.0
        self.b_squared = b**2.0
        sigma_squared = sigma**2.0
        sigma_delta_squared = 2.0 * sigma_squared  # the var of the differece is the sum of the vars
        self.sigma_2_delta_inv = 1.0 / sigma_delta_squared
        if use_approx:
            self.alpha = math.sqrt(1.0 + ((math.pi * sigma_delta_squared * self.b_squared) / 8.0))
        else:
            self.alpha = 1.0

        if update_method == 'batched':
            self.update = self.batched_update
        elif update_method == 'iterative':
            self.update = self.iterative_update

    def predict(self, matchups: np.ndarray, time_step: int = None, set_cache: bool = False):
        """
        Generates predictions for a series of matchups between competitors.
        """
        mus = self.mus[matchups]
        mu_diffs = mus[:, 0] - mus[:, 1]
        return sigmoid(self.b * mu_diffs / self.alpha)

    def get_pre_match_ratings(self, matchups: np.ndarray, **kwargs):
        return self.mus[matchups]

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
        for idx, (comp_1, comp_2) in enumerate(matchups):
            mu_1 = self.mus[comp_1]
            mu_2 = self.mus[comp_2]
            prob = sigmoid_scalar(self.b * (mu_1 - mu_2))
            k = self.b_over_2 / ((self.sigma_2_delta_inv) + (self.b_squared * prob * (1.0 - prob)))
            delta = k * (outcomes[idx] - prob)
            self.mus[comp_1] += delta
            self.mus[comp_2] -= delta

    def print_leaderboard(self, num_places):
        raise NotImplementedError
