"""TrueSkill"""
import math
import numpy as np
from scipy.stats import norm
from riix.core.base import OnlineRatingSystem
from riix.utils.math_utils import v_and_w_win_scalar, v_and_w_win_vector, v_and_w_draw_vector, v_and_w_draw_scalar


class TrueSkill(OnlineRatingSystem):
    """the og TrueSkill rating system shoutout to Microsoft"""

    rating_dim = 2

    def __init__(
        self,
        competitors: list,
        initial_mu: float = 25.0,
        initial_sigma: float = 8.333,
        beta: float = 4.166,
        tau: float = 0.0833,
        draw_probability=0.0,
        update_method: str = 'iterative',
        dtype=np.float64,
    ):
        super().__init__(competitors)
        self.beta = beta
        self.two_beta_squared = 2.0 * (beta**2.0)
        self.tau_squared = tau**2.0
        self.prev_time_step = 0

        self.epsilon = norm.ppf((draw_probability + 1.0) / 2.0) * math.sqrt(2.0) * beta
        self.mus = np.zeros(shape=self.num_competitors, dtype=dtype) + initial_mu
        self.sigma2s = np.zeros(shape=self.num_competitors, dtype=dtype) + initial_sigma**2.0
        self.has_played = np.zeros(shape=self.num_competitors, dtype=np.bool_)
        self.cache = {'combined_devs': None, 'norm_diffs': None}
        if update_method == 'batched':
            self.update = self.batched_update
        elif update_method == 'iterative':
            self.update = self.iterative_update

    @property
    def ratings(self):
        return self.mus

    def get_pre_match_ratings(self, matchups: np.ndarray, **kwargs):
        mus = self.mus[matchups]
        sigma2s = self.sigma2s[matchups]
        ratings = np.concatenate((mus[..., None], sigma2s[..., None]), axis=2).reshape(mus.shape[0], -1)
        return ratings

    def predict(self, matchups: np.ndarray, set_cache: bool = False, **kwargs):
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

    def increase_rating_dev(self, time_step, matchups):
        """called once per period to model the increase in variance over time"""
        active_in_period = np.unique(matchups)
        self.has_played[active_in_period] = True
        time_delta = time_step - self.prev_time_step
        self.sigma2s[self.has_played] += time_delta * self.tau_squared  # increase var for active players
        self.prev_time_step = time_step
        return active_in_period

    def batched_update(self, matchups, outcomes, time_step, use_cache=False, **kwargs):
        """apply one update based on all of the results of the rating period"""
        active_in_period = self.increase_rating_dev(time_step, matchups)
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

        vs = np.empty_like(norm_diffs)
        ws = np.empty_like(norm_diffs)
        win_mask = outcomes != 0.5
        draw_mask = ~win_mask

        vs[win_mask], ws[win_mask] = v_and_w_win_vector(
            norm_diffs[win_mask] * outcome_multiplier[win_mask], self.epsilon / combined_devs[win_mask]
        )
        vs[draw_mask], ws[draw_mask] = v_and_w_draw_vector(
            norm_diffs[draw_mask], self.epsilon / combined_devs[draw_mask]
        )

        mu_updates = vs[:, None] * (sigma2s / combined_devs[:, None])

        # map it back
        mu_updates[:, 1] *= -1.0
        mu_updates = mu_updates * outcome_multiplier[:, None]

        sigma2_updates = (np.square(sigma2s) * ws[:, None]) / combined_sigma2s[:, None]
        mu_updates_pooled = (mu_updates[:, :, None] * masks).sum((0, 1))

        matchups_per_competitor = np.sum(masks, axis=(0, 1))
        sigma2_updates_pooled = (sigma2_updates[:, :, None] * masks).sum(axis=(0, 1)) / matchups_per_competitor

        self.mus[active_in_period] += mu_updates_pooled
        self.sigma2s[active_in_period] -= sigma2_updates_pooled

    def iterative_update(self, matchups, outcomes, time_step, **kwargs):
        """treat the matchups in the rating period as if they were sequential"""
        self.increase_rating_dev(time_step, matchups)
        for idx in range(matchups.shape[0]):
            comp_1, comp_2 = matchups[idx]
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

            mu_updates = (sigma2s / combined_dev) * v * sign_multiplier
            sigma2_updates = (np.square(sigma2s) / combined_sigma2) * w
            self.mus[comp_1] += mu_updates[0]
            self.mus[comp_2] -= mu_updates[1]
            self.sigma2s[matchups[idx]] -= sigma2_updates

    def print_leaderboard(self, num_places):
        sort_array = self.mus - (3.0 * np.sqrt(self.sigma2s))
        sorted_idxs = np.argsort(-sort_array)[:num_places]
        max_len = min(np.max([len(comp) for comp in self.competitors] + [10]), 25)
        print(f'{"competitor": <{max_len}}\t{"mu - (3 * sd)"}')
        for p_idx in range(num_places):
            comp_idx = sorted_idxs[p_idx]
            print(f'{self.competitors[comp_idx]: <{max_len}}\t{sort_array[comp_idx]:.6f}')
