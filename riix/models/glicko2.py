"""
Glicko 2
paper: http://www.glicko.net/research/dpcmsv.pdf
example: http://www.glicko.net/glicko/glicko2.pdf

"""
import math
import numpy as np
from riix.core.base import OnlineRatingSystem
from riix.utils.math_utils import sigmoid, sigmoid_scalar
from riix.utils.constants import THREE_OVER_PI_SQUARED


class Glicko2(OnlineRatingSystem):
    """
    Implements the Glicko 2 rating system, designed by Mark Glickman.
    """

    rating_dim = 2

    def __init__(
        self,
        competitors: list,
        initial_rating: float = 0.0,  # I'm choosing the natural scale to avoid ocnverting back and forth
        initial_phi: float = 2.0147,
        initial_sigma: float = 0.06,
        tau: float = 0.2,
        dtype=np.float64,
        update_method='iterative',
    ):
        """Initializes the Glicko rating system with the given parameters."""
        super().__init__(competitors)
        self.mus = np.zeros(shape=self.num_competitors, dtype=dtype) + initial_rating
        self.initial_phi = initial_phi
        self.phis = np.zeros(shape=self.num_competitors, dtype=dtype) + initial_phi
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

    @staticmethod
    def g_vector(phi):
        """vector version"""
        return 1.0 / np.sqrt(1.0 + (THREE_OVER_PI_SQUARED * (phi**2.0)))

    # TODO should Glicko probs incorporate the dev increase?
    def predict(self, matchups: np.ndarray, time_step: int = None, set_cache: bool = False):
        """generate predictions"""
        mu_1 = self.mus[matchups[:, 0]]
        mu_2 = self.mus[matchups[:, 1]]
        rating_diffs = mu_1 - mu_2
        phi_1 = self.phis[matchups[:, 0]]
        phi_2 = self.phis[matchups[:, 1]]
        # the papers and theory seem to indicate this...
        # combined_dev = self.g_vector(np.sqrt(np.square(phi_1) + np.square(phi_2)))
        # but this seems to work better...
        combined_dev = self.g_vector(phi_1 + phi_2)
        probs = sigmoid(combined_dev * rating_diffs)
        return probs

    def get_pre_match_ratings(self, matchups: np.ndarray, **kwargs):
        means = self.mus[matchups]
        devs = self.phis[matchups]
        ratings = np.concatenate((means[..., None], devs[..., None]), axis=2).reshape(means.shape[0], -1)
        return ratings

    def increase_rating_dev(self, matchups):
        """called once per period to model the increase in variance over time"""
        active_in_period = np.unique(matchups)
        self.has_played[active_in_period] = True
        self.phis[self.has_played] = np.minimum(
            np.sqrt(np.square(self.phis[self.has_played]) + np.square(self.sigmas[self.has_played])), self.initial_phi
        )
        return active_in_period

    def batched_update(self, matchups, outcomes, use_cache=False):
        """apply one update based on all of the results of the rating period"""
        # TODO

    def get_sigma_star(self, sigma):
        return sigma

    def iterative_update(self, matchups, outcomes, **kwargs):
        """treat the matchups in the rating period as if they were sequential"""
        self.increase_rating_dev(matchups)
        for idx in range(matchups.shape[0]):
            comp_1, comp_2 = matchups[idx]
            mu_1 = self.mus[comp_1]
            mu_2 = self.mus[comp_2]
            rating_diff = mu_1 - mu_2
            phi_1 = self.phis[comp_1]
            phi_2 = self.phis[comp_2]
            g_1 = self.g_scalar(phi_1)
            g_2 = self.g_scalar(phi_2)
            p_1 = sigmoid_scalar(g_1 * rating_diff)
            p_2 = sigmoid_scalar(-g_2 * rating_diff)
            v_1 = 1.0 / ((g_1**2.0) * p_1 * (1.0 - p_1))
            v_2 = 1.0 / ((g_2**2.0) * p_2 * (1.0 - p_2))
            # delta_1 = v_1 * g_1 * (outcomes[idx] - p_1)
            # delta_2 = v_2 * g_2 * (1.0 - outcomes[idx] - p_2)

            # sigma_star_1 = self.get_sigma_star(self.sigmas[comp_1])
            # sigma_star_2 = self.get_sigma_star(self.sigmas[comp_2])

            # I guess don't do this since I update them all at the beginning?
            # phi_star_1 = (self.phis[comp_1] ** 2.0) + (sigma_star_1 ** 2.0)
            # phi_star_2 = (self.phis[comp_2] ** 2.0) + (sigma_star_2 ** 2.0)
            phi_star_1 = self.phis[comp_1] ** 2.0
            phi_star_2 = self.phis[comp_2] ** 2.0

            self.phis[comp_1] = 1.0 / math.sqrt((1.0 / phi_star_1) + (1.0 / (v_1**2.0)))
            self.phis[comp_2] = 1.0 / math.sqrt((1.0 / phi_star_2) + (1.0 / (v_2**2.0)))

            self.mus[comp_1] += (self.phis[comp_1] ** 2.0) * g_1 * (outcomes[idx] - p_1)
            self.mus[comp_2] += (self.phis[comp_2] ** 2.0) * g_2 * (1.0 - outcomes[idx] - p_2)

    def print_leaderboard(self, num_places):
        sort_array = self.ratings - (3.0 * self.rating_devs)
        sorted_idxs = np.argsort(-sort_array)[:num_places]
        max_len = min(np.max([len(comp) for comp in self.competitors] + [10]), 25)
        print(f'{"competitor": <{max_len}}\t{"rating - (3*dev)"}\t')
        for p_idx in range(num_places):
            comp_idx = sorted_idxs[p_idx]
            print(f'{self.competitors[comp_idx]: <{max_len}}\t{sort_array[comp_idx]:.6f}')
