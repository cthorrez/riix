"""
Online version of the Disc Decomposition rating system
https://proceedings.mlr.press/v206/bertrand23a.html
equations 8 and 9
"""
import numpy as np
from riix.core.base import OnlineRatingSystem
from riix.utils.math_utils import sigmoid


class OnlineDiscDecomp(OnlineRatingSystem):
    """Online Disc Decomposition"""

    rating_dim = 2

    def __init__(
        self,
        competitors: list,
        initial_u: float = 1.0,
        initial_v: float = 1.0,
        eta: float = 0.02,
        update_method: str = 'iterative',
        dtype=np.float64,
    ):
        super().__init__(competitors)
        self.initial_u = initial_u
        self.initial_v = initial_v
        self.eta = eta
        self.us = np.zeros(shape=self.num_competitors, dtype=dtype) + initial_u
        self.vs = np.zeros(shape=self.num_competitors, dtype=dtype) + initial_v
        self.cache = {'probs': None}
        if update_method == 'batched':
            self.update = self.batched_update
        elif update_method == 'iterative':
            self.update = self.iterative_update

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

    def batched_update(self, matchups, outcomes, use_cache, **kwargs):
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

    def print_leaderboard(self, num_places):
        sorted_idxs = np.argsort(-self.us)[:num_places]
        max_len = min(np.max([len(comp) for comp in self.competitors] + [10]), 25)
        print(f'{"competitor": <{max_len}}\t{"rating"}')
        for p_idx in range(num_places):
            comp_idx = sorted_idxs[p_idx]
            print(f'{self.competitors[comp_idx]: <{max_len}}\t{self.us[comp_idx]:.6f}')
