"""Weng/Lin Bayesian Online Rating system, Thurstone Mosteller Edition"""
import math
import numpy as np
from scipy.stats import norm
from riix.core.base import OnlineRatingSystem
from riix.utils.math_utils import v_and_w_win_scalar, v_and_w_draw_scalar


class WengLinThurstoneMosteller(OnlineRatingSystem):
    """The Bayesian Online Rating System introduced by Weng and Lin"""

    def __init__(
        self,
        num_competitors: int,
        initial_mu: float = 25.0,
        initial_sigma: float = 8.333,
        beta: float = 4.166,
        kappa: float = 0.0001,
        tau: float = 0.0833,
        draw_probability=0.0,
        update_method: str = 'iterative',
        dtype=np.float64,
    ):
        self.num_competitors = num_competitors
        self.beta = beta
        self.kappa = kappa
        self.two_beta_squared = 2.0 * (beta**2.0)
        self.tau_squared = tau**2.0
        self.epsilon = norm.ppf((draw_probability + 1.0) / 2.0) * math.sqrt(2.0) * beta

        self.mus = np.zeros(shape=num_competitors, dtype=dtype) + initial_mu
        self.sigma2s = np.zeros(shape=num_competitors, dtype=dtype) + initial_sigma**2.0
        self.has_played = np.zeros(shape=num_competitors, dtype=np.bool_)
        self.cache = {'TODO': None}

        if update_method == 'batched':
            self.update_fn = self.batched_update
        elif update_method == 'iterative':
            self.update_fn = self.iterative_update

    def predict(self, time_step: int, matchups: np.ndarray, set_cache: bool = False):
        """generate predictions"""
        mus = self.mus[matchups]
        sigma2s = self.sigma2s[matchups]
        combined_sigma2s = self.two_beta_squared + sigma2s.sum(axis=1)
        combined_devs = np.sqrt(combined_sigma2s)
        norm_diffs = (mus[:, 0] - mus[:, 1]) / combined_devs
        probs = norm.cdf(norm_diffs)
        return probs

    def fit(
        self,
        time_step: int,
        matchups: np.ndarray,
        outcomes: np.ndarray,
        use_cache: bool = False,
    ):
        self.update_fn(matchups, outcomes, use_cache=use_cache)

    def increase_rating_dev(self, matchups):
        """called once per period to model the increase in variance over time"""
        active_in_period = np.unique(matchups)
        self.has_played[active_in_period] = True
        # self.sigma2s[active_in_period] += self.tau_squared  # increase var for currently playing players
        self.sigma2s[self.has_played] += self.tau_squared  # increase var for ALL players
        return active_in_period

    def batched_update(self, matchups, outcomes, use_cache=False):
        """apply one update based on all of the results of the rating period"""
        pass

    def iterative_update(self, matchups, outcomes, **kwargs):
        """treat the matchups in the rating period as if they were sequential"""
        for idx in range(matchups.shape[0]):
            comp_1, comp_2 = matchups[idx]
            self.sigma2s[matchups[idx]] += self.tau_squared
            rating_diff = self.mus[comp_1] - self.mus[comp_2]
            sigma2s = self.sigma2s[matchups[idx]]

            combined_sigma2 = self.two_beta_squared + sigma2s.sum()
            combined_dev = np.sqrt(combined_sigma2)
            norm_diff = (rating_diff) / combined_dev

            outcome = outcomes[idx]
            sign_multiplier = outcome if outcome else -1.0
            if outcome != 0.5:
                v, w = v_and_w_win_scalar(norm_diff * sign_multiplier, self.epsilon / combined_dev)
            else:
                v, w = v_and_w_draw_scalar(norm_diff, self.epsilon / combined_dev)

            deltas = sign_multiplier * (sigma2s / combined_dev) * v

            gammas = np.sqrt(sigma2s) / combined_dev
            etas = gammas * (sigma2s / combined_sigma2) * w
            sigma2_multipliers = np.maximum(1.0 - etas, self.kappa)

            self.mus[comp_1] += deltas[0]
            self.mus[comp_2] -= deltas[1]
            self.sigma2s[matchups[idx]] *= sigma2_multipliers
