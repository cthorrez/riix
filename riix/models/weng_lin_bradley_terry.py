"""Weng/Lin Bayesian Online Rating system, Bradley Terry Edition"""
import numpy as np
from riix.core.base import OnlineRatingSystem
from riix.utils.math_utils import sigmoid


class WengLinBradleyTerry(OnlineRatingSystem):
    """The Bayesian Online Rating System introduced by Weng and Lin"""

    def __init__(
        self,
        num_competitors: int,
        initial_mu: float = 25.0,
        initial_sigma: float = 8.333,
        beta: float = 4.166,
        kappa: float = 0.0001,
        tau: float = 0.0833,
        update_method: str = 'iterative',
        dtype=np.float64,
    ):
        self.num_competitors = num_competitors
        self.beta = beta
        self.two_beta_squared = 2.0 * (beta**2.0)
        self.tau_squared = tau**2.0
        self.kappa = kappa

        self.mus = np.zeros(shape=num_competitors, dtype=dtype) + initial_mu
        self.sigma2s = np.zeros(shape=num_competitors, dtype=dtype) + initial_sigma**2.0
        self.has_played = np.zeros(shape=num_competitors, dtype=np.bool_)

        if update_method == 'batched':
            self.update = self.batched_update
        elif update_method == 'iterative':
            self.update = self.iterative_update

        self.cache = {
            'combined_sigma2s': None,
            'combined_devs': None,
            'probs': None,
        }

    def predict(self, time_step: int, matchups: np.ndarray, set_cache: bool = False):
        """generate predictions"""
        mus = self.mus[matchups]
        sigma2s = self.sigma2s[matchups]
        combined_sigma2s = self.two_beta_squared + sigma2s.sum(axis=1)
        combined_devs = np.sqrt(combined_sigma2s)
        norm_diffs = (mus[:, 0] - mus[:, 1]) / combined_devs
        probs = sigmoid(norm_diffs)
        if set_cache:
            self.cache['combined_sigma2s'] = combined_sigma2s
            self.cache['combined_devs'] = combined_devs
            self.cache['probs'] = probs
        return probs

    def fit(
        self,
        time_step: int,
        matchups: np.ndarray,
        outcomes: np.ndarray,
        use_cache: bool = False,
    ):
        self.update(matchups, outcomes, use_cache=use_cache)

    def increase_rating_dev(self, matchups):
        """called once per period to model the increase in variance over time"""
        active_in_period = np.unique(matchups)
        self.has_played[active_in_period] = True
        # self.sigma2s[active_in_period] += self.tau_squared  # increase var for currently playing players
        self.sigma2s[self.has_played] += self.tau_squared  # increase var for ALL players
        return active_in_period

    def batched_update(self, matchups, outcomes, use_cache=False):
        """apply one update based on all of the results of the rating period"""
        active_in_period = self.increase_rating_dev(matchups)
        masks = np.equal(matchups[:, :, None], active_in_period[None, :])  # N x 2 x active

        mus = self.mus[matchups]
        sigma2s = self.sigma2s[matchups]

        if use_cache:
            combined_sigma2s = self.cache['combined_sigma2s']
            combined_devs = self.cache['combined_devs']
            probs = self.cache['probs']
        else:
            combined_sigma2s = self.two_beta_squared + sigma2s.sum(axis=1)
            combined_devs = np.sqrt(combined_sigma2s)
            norm_diffs = (mus[:, 0] - mus[:, 1]) / combined_devs
            probs = sigmoid(norm_diffs)

        mu_updates = (sigma2s / combined_devs[:, None]) * (outcomes - probs)[:, None]
        mu_updates[:, 1] *= -1.0

        gammas = np.sqrt(sigma2s) / combined_devs[:, None]
        etas = gammas * (sigma2s / combined_sigma2s[:, None]) * (probs * (1.0 - probs))[:, None]

        sigma2_multipliers = np.maximum(1.0 - etas, self.kappa)

        mu_updates_pooled = (mu_updates[:, :, None] * masks).sum((0, 1))
        # aggregating by prod does NOT work well empirically, variance decreases way too fast
        # empirically max works really well
        sigma2_multipliers_pooled = np.max(sigma2_multipliers[:, :, None] * masks, axis=(0, 1))

        self.mus[active_in_period] += mu_updates_pooled
        self.sigma2s[active_in_period] *= sigma2_multipliers_pooled

    def iterative_update(self, matchups, outcomes, **kwargs):
        """treat the matchups in the rating period as if they were sequential"""
        for idx in range(matchups.shape[0]):
            comp_1, comp_2 = matchups[idx]
            sigma2s = self.sigma2s[matchups[idx]]
            self.sigma2s[matchups[idx]] += self.tau_squared
            combined_sigma2s = self.two_beta_squared + sigma2s.sum()
            combined_devs = np.sqrt(combined_sigma2s)
            norm_diffs = (self.mus[comp_1] - self.mus[comp_2]) / combined_devs
            prob = sigmoid(norm_diffs)

            mu_updates = (sigma2s / combined_devs) * (outcomes[idx] - prob)

            gammas = np.sqrt(sigma2s) / combined_devs
            etas = gammas * (sigma2s / combined_sigma2s) * (prob * (1.0 - prob))

            sigma2_multipliers = np.maximum(1.0 - etas, self.kappa)

            self.mus[comp_1] += mu_updates[0]
            self.mus[comp_2] -= mu_updates[1]
            self.sigma2s[matchups[idx]] *= sigma2_multipliers
