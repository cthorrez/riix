"""
Online version of the Disc Decomposition rating system
https://proceedings.mlr.press/v206/bertrand23a.html
equations 8 and 9
"""
import numpy as np
from riix.core.base import OnlineRatingSystem
from riix.utils.math_utils import sigmoid


class OnlineDiscDecomp(OnlineRatingSystem):
    def __init__(
        self,
        num_competitors: int,
        initial_u: float = 1.0,
        initial_v: float = 1.0,
        eta: float = 0.02,
        update_method: str = 'iterative',
        dtype=np.float64,
    ):
        self.num_competitors = num_competitors
        self.initial_u = initial_u
        self.initial_v = initial_v
        self.eta = eta
        self.us = np.zeros(shape=num_competitors, dtype=dtype) + initial_u
        self.vs = np.zeros(shape=num_competitors, dtype=dtype) + initial_v
        self.cache = {'probs': None}
        if update_method == 'batched':
            self.update_fn = self.batched_update
        elif update_method == 'iterative':
            self.update_fn = self.iterative_update

    def predict(self, time_step: int, matchups: np.ndarray, set_cache: bool = False):
        """generate predictions"""
        us_1 = self.us[matchups[:, 0]]
        us_2 = self.us[matchups[:, 1]]
        vs_1 = self.vs[matchups[:, 0]]
        vs_2 = self.vs[matchups[:, 1]]
        probs = sigmoid(vs_1 * vs_2 * (us_1 - us_2))
        if set_cache:
            self.cache['probs'] = probs
        return probs

    def fit(
        self,
        time_step: int,
        matchups: np.ndarray,
        outcomes: np.ndarray,
        use_cache: bool = False,
    ):
        self.update_fn(matchups, outcomes, use_cache=use_cache)

    def batched_update(self, matchups, outcomes, use_cache):
        """apply one update based on all of the results of the rating period"""
        active_in_period = np.unique(matchups)
        masks = np.equal(matchups[:, :, None], active_in_period[None, :])  # N x 2 x active
        if use_cache:
            probs = self.cache['probs']
        else:
            probs = self.predict(time_step=None, matchups=matchups, set_cache=False)
        per_match_diff = (outcomes - probs)[:, None]
        per_match_diff = np.hstack([per_match_diff, -per_match_diff])
        per_competitor_diff = (per_match_diff[:, :, None] * masks).sum(axis=(0, 1))
        self.ratings[active_in_period] += self.k * per_competitor_diff

    def iterative_update(self, matchups, outcomes, **kwargs):
        """treat the matchups in the rating period as if they were sequential"""
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
            self.us[comp_1] += v_update * v_2
            self.us[comp_2] -= v_update * v_1

    def print_top_k(self, k, competitor_names):
        sorted_idxs = np.argsort(-self.us)[:k]
        for k_idx in range(k):
            comp_idx = sorted_idxs[k_idx]
            print(competitor_names[comp_idx], self.us[comp_idx])