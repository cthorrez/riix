"""Weng/Lin Bayesian Online Rating system"""
import math
import numpy as np
from scipy.stats import norm
from riix.core.base import OnlineRatingSystem
from riix.utils.math_utils import sigmoid, sigmoid_scalar
from riix.utils.math_utils import v_and_w_win_scalar, v_and_w_draw_scalar


class WengLin(OnlineRatingSystem):
    """The Bayesian Online Rating System introduced by Weng and Lin"""

    rating_dim = 2

    def __init__(
        self,
        competitors: list,
        initial_mu: float = 25.0,
        initial_sigma: float = 8.333,
        beta: float = 4.166,
        kappa: float = 0.0001,
        tau: float = 0.0833,
        draw_probability=0.0,
        update_method: str = 'iterative',
        model='tm',
        dtype=np.float64,
    ):
        super().__init__(competitors)
        self.beta = beta
        self.two_beta_squared = 2.0 * (beta**2.0)
        self.tau_squared = tau**2.0
        self.kappa = kappa
        self.epsilon = norm.ppf((draw_probability + 1.0) / 2.0) * math.sqrt(2.0) * beta

        self.mus = np.zeros(shape=self.num_competitors, dtype=dtype) + initial_mu
        self.sigma2s = np.zeros(shape=self.num_competitors, dtype=dtype) + initial_sigma**2.0
        self.has_played = np.zeros(shape=self.num_competitors, dtype=np.bool_)
        self.prev_time_step = 0

        if update_method == 'batched':
            self.update = self.batched_update
        elif update_method == 'iterative':
            self.update = self.iterative_update

        if model == 'bt':
            self.model_func = self.bradley_terry_scalar_updates
            self.prob_func = sigmoid
        elif model == 'tm':
            self.model_func = self.thurstone_mosteller_scalar_updates
            self.prob_func = norm.cdf

        self.cache = {
            'combined_sigma2s': None,
            'combined_devs': None,
            'probs': None,
        }

    def get_pre_match_ratings(self, matchups: np.ndarray, **kwargs):
        mus = self.mus[matchups]
        sigma2s = self.sigma2s[matchups]
        ratings = np.concatenate((mus[..., None], sigma2s[..., None]), axis=2).reshape(mus.shape[0], -1)
        return ratings

    def predict(self, matchups: np.ndarray, time_step: int = None, set_cache: bool = False):
        """generate predictions"""
        # TODO should add time based var increase to prediction logic
        mus = self.mus[matchups]
        sigma2s = self.sigma2s[matchups]
        combined_sigma2s = self.two_beta_squared + sigma2s.sum(axis=1)
        combined_devs = np.sqrt(combined_sigma2s)
        norm_diffs = (mus[:, 0] - mus[:, 1]) / combined_devs
        probs = self.prob_func(norm_diffs)
        if set_cache:
            self.cache['combined_sigma2s'] = combined_sigma2s
            self.cache['combined_devs'] = combined_devs
            self.cache['probs'] = probs
        return probs

    def increase_rating_dev(self, time_step, matchups):
        """called once per period to model the increase in variance over time"""
        active_in_period = np.unique(matchups)
        self.has_played[active_in_period] = True
        time_delta = time_step - self.prev_time_step
        self.sigma2s[self.has_played] += time_delta * self.tau_squared  # increase var for active players
        self.prev_time_step = time_step
        return active_in_period

    def batched_update(self, matchups, outcomes, use_cache=False, **kwargs):
        """apply one update based on all of the results of the rating period"""
        # TODO fix the update to actually be based on time...
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

    def bradley_terry_scalar_updates(self, norm_diff, sigma2s, combined_sigma2, combined_dev, outcome):
        prob = sigmoid_scalar(norm_diff)
        deltas = (sigma2s / combined_dev) * (outcome - prob)
        gammas = np.sqrt(sigma2s) / combined_dev
        etas = gammas * (sigma2s / combined_sigma2) * (prob * (1.0 - prob))
        return deltas, etas

    def thurstone_mosteller_scalar_updates(self, norm_diff, sigma2s, combined_sigma2, combined_dev, outcome):
        sign_multiplier = outcome if outcome else -1.0
        if outcome != 0.5:
            v, w = v_and_w_win_scalar(norm_diff * sign_multiplier, self.epsilon / combined_dev)
        else:
            v, w = v_and_w_draw_scalar(norm_diff, self.epsilon / combined_dev)

        deltas = sign_multiplier * (sigma2s / combined_dev) * v
        gammas = np.sqrt(sigma2s) / combined_dev
        etas = gammas * (sigma2s / combined_sigma2) * w
        return deltas, etas

    def iterative_update(self, matchups, outcomes, time_step, **kwargs):
        """treat the matchups in the rating period as if they were sequential"""
        self.increase_rating_dev(time_step, matchups)
        for idx in range(matchups.shape[0]):
            comp_1, comp_2 = matchups[idx]
            # self.sigma2s[matchups[idx]] += self.tau_squared
            sigma2s = self.sigma2s[matchups[idx]]
            combined_sigma2 = self.two_beta_squared + sigma2s.sum()
            combined_dev = np.sqrt(combined_sigma2)
            norm_diff = (self.mus[comp_1] - self.mus[comp_2]) / combined_dev

            deltas, etas = self.model_func(norm_diff, sigma2s, combined_sigma2, combined_dev, outcomes[idx])
            sigma2_multipliers = np.maximum(1.0 - etas, self.kappa)

            self.mus[comp_1] += deltas[0]
            self.mus[comp_2] -= deltas[1]
            self.sigma2s[matchups[idx]] *= sigma2_multipliers

    def print_leaderboard(self, num_places):
        sort_array = self.mus - (3.0 * np.sqrt(self.sigma2s))
        sorted_idxs = np.argsort(-sort_array)[:num_places]
        max_len = min(np.max([len(comp) for comp in self.competitors] + [10]), 25)
        print(f'{"competitor": <{max_len}}\t{"mu - (3 * sd)"}')
        for p_idx in range(num_places):
            comp_idx = sorted_idxs[p_idx]
            print(f'{self.competitors[comp_idx]: <{max_len}}\t{sort_array[comp_idx]:.6f}')
