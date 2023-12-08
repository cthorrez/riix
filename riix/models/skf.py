"""simplified kalman filter"""
import math
import numpy as np
from riix.core.base import OnlineRatingSystem
from riix.utils.math_utils import base_10_sigmoid

LOG10 = math.log(10.0)
LOG10_SQUARED = LOG10**2.0


class VSKF(OnlineRatingSystem):
    """vector covariance simplified kalman filter"""

    def __init__(
        self,
        num_competitors: int,
        mu_0: float = 0.0,
        v_0: float = 1.0,
        beta: float = 1.0,
        s: float = 1.0,
        epsilon: float = 1e-3,
        dtype=np.float64,
        update_method='iterative',
    ):
        self.num_competitors = num_competitors
        self.mus = np.zeros(num_competitors, dtype=dtype) + mu_0
        self.vs = np.zeros(num_competitors, dtype=dtype) + v_0
        self.has_played = np.zeros(num_competitors, dtype=np.bool_)
        self.beta = beta
        self.beta2 = beta**2.0
        self.s = s
        self.s2 = s**2.0
        self.epsilon = epsilon
        self.prev_time_step = 0

        if update_method == 'batched':
            self.update_function = self.batched_update
        elif update_method == 'iterative':
            self.update_function = self.iterative_update

    def predict(self, time_step: int, matchups: np.ndarray, set_cache: bool = False):
        """generate predictions"""
        ratings_1 = self.mus[matchups[:, 0]]
        ratings_2 = self.mus[matchups[:, 1]]
        rating_diffs = ratings_1 - ratings_2
        probs = base_10_sigmoid(rating_diffs / self.s)
        return probs

    def fit(
        self,
        time_step: int,
        matchups: np.ndarray,
        outcomes: np.ndarray,
        use_cache: bool = False,
    ):
        self.update_function(time_step, matchups, outcomes, use_cache=use_cache)

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

    def batched_update(self, time_step, matchups, outcomes, use_cache=False):
        """apply one update based on all of the results of the rating period"""
        pass

    def iterative_update(self, time_step, matchups, outcomes, **kwargs):
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
            prob = base_10_sigmoid(z)

            g = LOG10 * (outcome - prob)
            h = LOG10_SQUARED * prob * (1.0 - prob)

            denom = (self.s2) + h * omega
            mu_update = (self.s * g) / denom

            v_update = h / denom

            self.mus[comp_1] += v_1 * mu_update
            self.mus[comp_2] -= v_2 * mu_update
            self.vs[comp_1] *= 1.0 - v_1 * v_update
            self.vs[comp_2] *= 1.0 - v_2 * v_update
