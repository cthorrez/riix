"""
Elo + Momentum
Invented by Clayton Thorrez (me)
"""
import math
import numpy as np
from riix.core.base import OnlineRatingSystem
from riix.models.elo import Elo
from riix.utils.math_utils import sigmoid


class EloMentum(Elo):
    """Elo with momentum!"""

    rating_dim = 1

    def __init__(
        self,
        competitors: list,
        initial_rating: float = 1500.0,
        k: float = 32.0,
        alpha: float = math.log(10.0) / 400.0,
        momentum: float = 0.2,
        # momentum=(0.2, 0.9),
        momentum_type: str = 'nesterov',
        update_method: str = 'iterative',
        epsilon: float = 1e-8,
        dtype=np.float64,
    ):
        OnlineRatingSystem.__init__(self, competitors)
        self.k = k
        self.alpha = alpha
        self.momentum = momentum
        self.epsilon = epsilon
        self.ratings = np.zeros(shape=self.num_competitors, dtype=dtype) + initial_rating

        self.v = np.zeros(shape=self.num_competitors, dtype=dtype)
        self.cache = {'probs': None}
        if update_method == 'batched':
            self.update = self.batched_update
        elif update_method == 'iterative':
            self.update = self.iterative_update

        if momentum_type == 'heavy_ball':
            self.momentum_fn = self.get_momentum_update
        elif momentum_type == 'nesterov':
            self.momentum_fn = self.get_nesterov_momentum_update
        elif momentum_type == 'adam':
            self.momentum_fn = self.get_adam_update
            self.mu = np.zeros(shape=self.num_competitors, dtype=dtype)
            self.t = np.zeros(shape=self.num_competitors, dtype=np.int32)
            self.beta1, self.beta2 = momentum
        else:
            raise ValueError(f'Invalid momentum_type {momentum_type}')

    def get_pre_match_ratings(self, matchups: np.ndarray, **kwargs):
        return self.ratings[matchups]

    def predict(self, time_step: int, matchups: np.ndarray, set_cache: bool = False):
        """generate predictions"""
        ratings_1 = self.ratings[matchups[:, 0]]
        ratings_2 = self.ratings[matchups[:, 1]]
        probs = sigmoid(self.alpha * (ratings_1 - ratings_2))
        if set_cache:
            self.cache['probs'] = probs
        return probs

    def update(self, matchups: np.ndarray, outcomes: np.ndarray, use_cache: bool = False, **kwargs):
        self.update(matchups, outcomes, use_cache=use_cache)

    def get_momentum_update(self, idx, g):
        v = self.v[idx]
        v_new = (self.momentum * v) + g
        self.v[idx] = v_new
        update = self.k * v_new
        return update

    def get_nesterov_momentum_update(self, idx, g):
        # from https://cs231n.github.io/neural-networks-3/#sgd
        # v_prev = v # back this up
        # v = mu * v - learning_rate * dx # velocity update stays the same
        # x += -mu * v_prev + (1 + mu) * v # position update changes form
        v = self.v[idx]
        v_new = (self.momentum * v) + (self.k * g)
        update = (self.momentum * v) + ((1.0 + self.momentum) * v_new)
        self.v[idx] = v_new
        return update

    def get_adam_update(self, idx, g):
        mu = self.mu[idx]
        v = self.v[idx]
        t = self.t[idx]
        t += 1.0
        mu_new = (self.beta1 * mu) + ((1.0 - self.beta1) * g)
        v_new = (self.beta2 * v) + ((1.0 - self.beta2) * (g * g))
        mu_hat = mu_new / (1.0 - (self.beta1**t))
        v_hat = v_new / (1.0 - (self.beta2**t))
        update = self.k * mu_hat / (math.sqrt(v_hat) + self.epsilon)
        self.mu[idx] = mu_new
        self.v[idx] = v_new
        self.t[idx] = t
        return update

    def batched_update(self, matchups, outcomes, use_cache):
        """apply one update based on all of the results of the rating period"""
        pass
        # active_in_period = np.unique(matchups)
        # masks = np.equal(matchups[:, :, None], active_in_period[None, :])  # N x 2 x active
        # if use_cache:
        #     probs = self.cache['probs']
        # else:
        #     probs = self.predict(time_step=None, matchups=matchups, set_cache=False)
        # per_match_diff = (outcomes - probs)[:, None]
        # per_match_diff = np.hstack([per_match_diff, -per_match_diff])
        # per_competitor_diff = (per_match_diff[:, :, None] * masks).sum(axis=(0, 1))
        # self.ratings[active_in_period] += self.k * per_competitor_diff

    def iterative_update(self, matchups, outcomes, **kwargs):
        """treat the matchups in the rating period as if they were sequential"""
        for idx in range(matchups.shape[0]):
            comp_1, comp_2 = matchups[idx]
            diff = self.ratings[comp_1] - self.ratings[comp_2]
            prob = sigmoid(self.alpha * diff)
            # I think technically this is not the gradient because of alpha but it's close enough
            g_1 = outcomes[idx] - prob
            g_2 = -1.0 * g_1
            update_1 = self.momentum_fn(comp_1, g_1)
            update_2 = self.momentum_fn(comp_2, g_2)
            self.ratings[comp_1] += update_1
            self.ratings[comp_2] += update_2
