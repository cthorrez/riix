"""
Online version of the Disc Decomposition rating system
https://proceedings.mlr.press/v206/bertrand23a.html
equations 8 and 9
"""

import numpy as np
from riix.core.base import OnlineRatingSystem
from riix.utils.math_utils import sigmoid, scatter_sum


class OnlineDiscDecomp(OnlineRatingSystem):
    """Online Disc Decomposition"""

    rating_dim = 2

    def __init__(
        self,
        competitors: list,
        initial_u: float = 1.0,
        initial_v: float = 1.0,
        eta: float = 0.02,
        update_method: str = 'online',
        dtype=np.float64,
    ):
        super().__init__(competitors, update_method)
        self.initial_u = initial_u
        self.initial_v = initial_v
        self.eta = eta
        self.us = np.zeros(shape=self.num_competitors, dtype=dtype) + initial_u
        self.vs = np.zeros(shape=self.num_competitors, dtype=dtype) + initial_v
        self.cache = {'probs': None}

    def get_pre_match_ratings(self, matchups: np.ndarray, **kwargs):
        us = self.us[matchups]
        vs = self.vs[matchups]
        ratings = np.concatenate((us[..., None], vs[..., None]), axis=2).reshape(us.shape[0], -1)
        return ratings

    def predict(self, matchups: np.ndarray, set_cache: bool = False, **kwargs):
        us_1 = self.us[matchups[:, 0]]
        us_2 = self.us[matchups[:, 1]]
        vs_1 = self.vs[matchups[:, 0]]
        vs_2 = self.vs[matchups[:, 1]]
        probs = sigmoid(vs_1 * vs_2 * (us_1 - us_2))
        if set_cache:
            self.cache['probs'] = probs
        return probs

    def batched_update(self, matchups, outcomes, use_cache=False, **kwargs):
        """apply one update based on all of the results of the rating period"""
        active_in_period = np.unique(matchups)
        us = self.us[matchups]
        vs = self.vs[matchups]
        u_diffs = us[:, 0] - us[:, 1]
        v_prods = vs[:, 0] * vs[:, 1]
        if use_cache:
            probs = self.cache['probs']
        else:
            probs = sigmoid(v_prods * u_diffs)
        scaled_deltas = self.eta * (outcomes - probs)

        # paper equation 8: u_i += eta * delta * v_i * v_j, u_j gets the negated update
        u_updates = (scaled_deltas * v_prods)[:, None] * np.array([1.0, -1.0])
        # paper equation 9: v_i += eta * delta * v_j * (u_i - u_j), same sign for both sides
        v_updates = (scaled_deltas * u_diffs)[:, None] * vs[:, [1, 0]]

        self.us[active_in_period] += scatter_sum(u_updates, matchups, self.num_competitors)[active_in_period]
        self.vs[active_in_period] += scatter_sum(v_updates, matchups, self.num_competitors)[active_in_period]

    def online_update(self, matchups, outcomes, **kwargs):
        for idx in range(matchups.shape[0]):
            comp_1, comp_2 = matchups[idx]
            u_1, u_2 = self.us[matchups[idx]]
            v_1, v_2 = self.vs[matchups[idx]]
            u_diff = u_1 - u_2
            v_prod = v_1 * v_2
            prob = sigmoid(v_prod * u_diff)
            scaled_delta = self.eta * (outcomes[idx] - prob)

            u_update = scaled_delta * v_prod
            v_update = scaled_delta * u_diff

            self.us[comp_1] += u_update
            self.us[comp_2] -= u_update
            self.vs[comp_1] += v_update * v_2
            self.vs[comp_2] += v_update * v_1

    def leaderboard_ratings(self):
        return self.us
