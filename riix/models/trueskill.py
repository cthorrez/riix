"""TrueSkill"""
import numpy as np
from scipy.stats import norm
from riix.core.base import OnlineRatingSystem
from riix.utils.math_utils import norm_pdf, norm_cdf


def v_and_w_vector(t, eps):
    """calculate v and w for a win"""
    diff = t - eps
    v = norm.pdf(diff) / norm.cdf(diff)

    bad_mask = np.isnan(v) | np.isinf(v)
    if bad_mask.any():
        v[bad_mask] = (-1 * (diff))[bad_mask]
    w = v * (v + diff)
    return v, w


def v_and_w_scalar(t, eps):
    diff = t - eps
    try:
        v = norm_pdf(diff) / norm_cdf(diff)
    except ZeroDivisionError:
        v = -diff
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
        draw_probability=0.0,
        update_method: str = 'batched',
        dtype=np.float64,
    ):
        self.num_competitors = num_competitors
        self.beta = beta
        self.two_beta_squared = 2.0 * (beta**2.0)
        self.tau_squared = tau**2.0

        self.epsilon = norm.ppf((draw_probability + 1.0) / 2.0) * np.sqrt(2.0) * beta
        self.mus = np.zeros(shape=num_competitors, dtype=dtype) + initial_mu
        self.sigma2s = np.zeros(shape=num_competitors, dtype=dtype) + initial_sigma**2.0
        self.has_played = np.zeros(shape=num_competitors, dtype=np.bool_)
        self.cache = {'combined_devs': None, 'norm_diffs': None}
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
    ):
        self.update_fn(matchups, outcomes, use_cache=use_cache)

    def increase_rating_dev(self, matchups):
        """called once per period to model the increase in variance over time"""
        active_in_period = np.unique(matchups)
        self.has_played[active_in_period] = True
        self.sigma2s[active_in_period] += self.tau_squared  # increase var for currently playing players
        # self.sigma2s[self.has_played] += self.tau_squared  # increase var for ALL players
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
        outcome_multiplier[outcome_multiplier == 0] = -1.0

        vs, ws = v_and_w_vector(norm_diffs * outcome_multiplier, self.epsilon / combined_devs)

        mu_updates = vs[:, None] * (sigma2s / combined_devs[:, None])

        # map it back?
        mu_updates[:, 1] *= -1.0
        mu_updates = mu_updates * outcome_multiplier[:, None]

        sigma2_updates = (np.square(sigma2s) * ws[:, None]) / combined_sigma2s[:, None]
        mu_updates_pooled = (mu_updates[:, :, None] * masks).sum((0, 1))
        sigma2_updates_pooled = (sigma2_updates[:, :, None] * masks).mean((0, 1))

        self.mus[active_in_period] += mu_updates_pooled
        self.sigma2s[active_in_period] -= sigma2_updates_pooled

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
            outcome_multiplier = outcome if outcome else -1.0
            v, w = v_and_w_scalar(norm_diff * outcome_multiplier, self.epsilon / combined_dev)

            sign_multiplier = 1.0 if outcome == 1 else -1
            mu_updates = (sigma2s / combined_dev) * v * sign_multiplier

            sigma2_updates = (np.square(sigma2s) / combined_sigma2) * w
            self.mus[comp_1] += mu_updates[0]
            self.mus[comp_2] -= mu_updates[1]
            self.sigma2s[matchups[idx]] -= sigma2_updates
