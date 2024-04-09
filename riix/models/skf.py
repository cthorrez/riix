"""simplified kalman filter"""
import math
import numpy as np
from scipy.stats import norm
from riix.core.base import OnlineRatingSystem
from riix.utils.math_utils import base_10_sigmoid
from riix.utils.math_utils import v_and_w_win_scalar, v_and_w_draw_scalar, norm_cdf

LOG10 = math.log(10.0)
LOG10_SQUARED = LOG10**2.0


def bradley_terry_prob(z):
    return base_10_sigmoid(z)


def bradley_terry_scalar_prob_g_h(z, outcome):
    prob = base_10_sigmoid(z)
    g = LOG10 * (outcome - prob)
    h = LOG10_SQUARED * prob * (1.0 - prob)
    return prob, g, h


def thurstone_mosteller_scalar_prob_g_h(z, outcome):
    prob = norm_cdf(z)
    if outcome != 0.5:
        sign_multiplier = (2 * outcome) - 1  # maps 1 to 1 and 0 to -1
        v, w = v_and_w_win_scalar(z * sign_multiplier, 0.0)
        v = v * sign_multiplier
    else:
        v, w = v_and_w_draw_scalar(z, 0.0)
    return prob, v, w


class VSKF(OnlineRatingSystem):
    """vector covariance simplified kalman filter"""

    rating_dim = 2

    def __init__(
        self,
        competitors: list,
        mu_0: float = 0.0,
        v_0: float = 1.0,
        beta: float = 1.0,
        s: float = 1.0,
        epsilon: float = 1e-2,
        dtype=np.float64,
        model='bt',  # legal values are bt, tm, and dd
        update_method='iterative',
    ):
        super().__init__(competitors)
        self.mus = np.zeros(self.num_competitors, dtype=dtype) + mu_0
        self.vs = np.zeros(self.num_competitors, dtype=dtype) + v_0
        self.has_played = np.zeros(self.num_competitors, dtype=np.bool_)
        self.beta = beta
        self.beta2 = beta**2.0
        self.s = s
        self.s2 = s**2.0
        self.epsilon = epsilon
        self.prev_time_step = 0

        if update_method == 'batched':
            self.update = self.batched_update
        elif update_method == 'iterative':
            self.update = self.iterative_update

        if model == 'bt':
            self.predict_func = bradley_terry_prob
            self.prob_g_h_func = bradley_terry_scalar_prob_g_h
        elif model == 'tm':
            self.predict_func = norm.cdf
            self.prob_g_h_func = thurstone_mosteller_scalar_prob_g_h

    def predict(self, matchups: np.ndarray, set_cache: bool = False, **kwargs):
        """generate predictions"""
        ratings_1 = self.mus[matchups[:, 0]]
        ratings_2 = self.mus[matchups[:, 1]]
        rating_diffs = ratings_1 - ratings_2
        probs = self.predict_func(rating_diffs / self.s)
        return probs

    @property
    def ratings(self):
        return self.mus

    def get_pre_match_ratings(self, matchups: np.ndarray, **kwargs):
        mus = self.mus[matchups]
        vs = self.vs[matchups]
        ratings = np.concatenate((mus[..., None], vs[..., None]), axis=2).reshape(mus.shape[0], -1)
        return ratings

    def time_dynamics_update(self, time_step, matchups):
        """called once per period to model the increase in variance over time"""
        active_in_period = np.unique(matchups)
        self.has_played[active_in_period] = True
        time_delta = time_step - self.prev_time_step

        # update parameters for passage of time
        beta_t = self.beta**time_delta
        self.mus[self.has_played] = beta_t * self.mus[self.has_played]
        self.vs[self.has_played] = (beta_t**2) * self.vs[self.has_played] + (time_delta * self.epsilon)
        self.prev_time_step = time_step

        return active_in_period

    def batched_update(self, matchups, outcomes, time_step, use_cache=False):
        """apply one update based on all of the results of the rating period"""
        active_in_period = self.time_dynamics_update(time_step, matchups)
        masks = np.equal(matchups[:, :, None], active_in_period[None, :])  # N x 2 x active

        mus = self.mus[matchups]
        vs = self.vs[matchups]

        omegas = np.sum(vs, axis=1)
        probs = base_10_sigmoid((mus[:, 0] - mus[:, 1]) / self.s)

        g = LOG10 * (outcomes - probs)
        h = LOG10_SQUARED * probs * (1.0 - probs)

        denom = (self.s2) + h * omegas
        mu_updates = (self.s * g) / denom
        v_updates = h / denom

        mu_updates = mu_updates[:, None].repeat(2, 1)
        mu_updates[:, 1] *= -1

        v_updates = 1.0 - vs * v_updates[:, None]

        mu_updates_pooled = (mu_updates[:, :, None] * masks).sum(axis=(0, 1))
        v_updates_pooled = (v_updates[:, :, None] * masks).max(axis=(0, 1))

        self.mus[active_in_period] += self.vs[active_in_period] * mu_updates_pooled
        self.vs[active_in_period] *= v_updates_pooled

    def iterative_update(self, matchups, outcomes, time_step, **kwargs):
        """treat the matchups in the rating period as if they were sequential"""
        self.time_dynamics_update(time_step, matchups)
        for idx in range(matchups.shape[0]):
            comp_1, comp_2 = matchups[idx]
            mu_1 = self.mus[comp_1]
            mu_2 = self.mus[comp_2]
            v_1 = self.vs[comp_1]
            v_2 = self.vs[comp_2]
            outcome = outcomes[idx]

            omega = v_1 + v_2
            z = (mu_1 - mu_2) / self.s

            prob, g, h = self.prob_g_h_func(z, outcome)

            denom = (self.s2) + h * omega
            mu_update = (self.s * g) / denom

            v_update = h / denom

            self.mus[comp_1] += v_1 * mu_update
            self.mus[comp_2] -= v_2 * mu_update
            self.vs[comp_1] *= 1.0 - v_1 * v_update
            self.vs[comp_2] *= 1.0 - v_2 * v_update

    def print_leaderboard(self, num_places):
        sort_array = self.mus - (3.0 * np.sqrt(self.vs))
        sorted_idxs = np.argsort(-sort_array)[:num_places]
        max_len = min(np.max([len(comp) for comp in self.competitors] + [10]), 25)
        print(f'{"competitor": <{max_len}}\t{"mu - (3 * sd)"}')
        for p_idx in range(num_places):
            comp_idx = sorted_idxs[p_idx]
            print(f'{self.competitors[comp_idx]: <{max_len}}\t{sort_array[comp_idx]:.6f}')
