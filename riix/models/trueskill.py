"""TrueSkill"""
import numpy as np
from scipy.stats import norm
from riix.core.base import OnlineRatingSystem


def v_and_w(t, eps):
    """calculate v and w in a pretty efficient way"""
    diff = t - eps
    v = norm.pdf(diff) / norm.cdf(diff)
    w = v * (v + diff)
    return v, w


class TrueSkill(OnlineRatingSystem):
    """the og TrueSkill rating system shoutout to Microsoft"""

    def __init__(
        self,
        num_competitors: int,
        initial_mu: float = 25.0,
        initial_sigma: float = 8.333,
        beta: float = 4.166,
        tau: float = 0.0833,
        epsilon: float = 0.0,
        dtype=np.float64,
    ):
        self.num_competitors = num_competitors
        self.beta = beta
        self.two_beta_squared = 2.0 * (beta**2.0)
        self.tau = tau
        self.tau_squared = tau**2.0
        self.epsilon = epsilon
        self.mus = np.zeros(shape=num_competitors, dtype=dtype) + initial_mu
        self.sigma2s = np.zeros(shape=num_competitors, dtype=dtype) + initial_sigma**2.0
        self.has_played = np.zeros(shape=num_competitors, dtype=np.bool_)
        self.cache = {'combined_devs': None, 'norm_diffs': None}

    def predict(self, time_step: int, matchups: np.ndarray, set_cache: bool = False):
        """generate predictions"""
        mus_1 = self.mus[matchups[:, 0]]
        mus_2 = self.mus[matchups[:, 1]]
        sigma2s_1 = self.sigma2s[matchups[:, 0]]
        sigma2s_2 = self.sigma2s[matchups[:, 1]]
        combined_sigma2s = self.two_beta_squared + sigma2s_1 + sigma2s_2
        combined_devs = np.sqrt(combined_sigma2s)
        norm_diffs = (mus_1 - mus_2) / combined_devs
        if set_cache:
            self.cache['norm_diffs'] = norm_diffs
            self.cache['combined_sigma2s'] = combined_sigma2s
            self.cache['combined_devs'] = combined_devs
        probs = norm.cdf(norm_diffs)
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
        self.sigma2s[self.has_played] += self.tau_squared
        return active_in_period

    def batched_update(self, matchups, outcomes, use_cache=False):
        """apply one update based on all of the results of the rating period"""
        active_in_period = self.increase_rating_dev(matchups)
        masks = np.equal(matchups[:, :, None], active_in_period[None, :])  # N x 2 x active

        sigma2s = self.sigma2s[matchups]
        if use_cache:
            norm_diffs = self.cache['norm_diffs']
            combined_devs = self.cache['combined_devs']
            combined_sigma2s = self.cache['combined_sigma2s']
        else:
            mus = self.mus[matchups]
            rating_diffs = mus[:, 0] - mus[:, 1]
            combined_sigma2s = self.two_beta_squared + sigma2s.sum(axis=1)
            combined_devs = np.sqrt(combined_sigma2s)
            norm_diffs = (rating_diffs) / combined_devs

        # map 1 to 1, 0 to -1, 0.5 to 0.5
        outcome_multiplier = outcomes.copy()
        outcome_multiplier[outcome_multiplier == 0] = -1
        outcome_multiplier = np.hstack([outcome_multiplier[:, None], -1 * outcome_multiplier[:, None]])

        vs, ws = v_and_w(norm_diffs[:, None] * outcome_multiplier, (self.epsilon / combined_devs)[:, None])

        mu_updates = vs[:, None] * (sigma2s / combined_devs[:, None])
        sigma2_updates = ws[:, None] * sigma2s / combined_sigma2s[:, None]
        mu_updates_pooled = (mu_updates[:, :, None] * masks).sum((0, 1))
        sigma2_updates_pooled = (sigma2_updates[:, :, None] * masks).sum((0, 1))

        self.mus[active_in_period] += mu_updates_pooled
        self.sigma2s[active_in_period] -= sigma2_updates_pooled

    def iterative_update(self, matchups, outcomes):
        """treat the matchups in the rating period as if they were sequential"""
        self.increase_rating_dev(matchups)
        for idx in range(matchups.shape[0]):
            comp_1, comp_2 = matchups[idx]
            rating_diff = self.mus[comp_1] - self.mus[comp_2]
            print(rating_diff)
