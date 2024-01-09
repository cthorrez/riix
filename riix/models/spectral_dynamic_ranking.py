"""
A Spectral Approach for the Dynamic Bradley-Terry Model
https://arxiv.org/abs/2307.16642
"""
import numpy as np
from riix.core.base import OnlineRatingSystem


class SpectralDynamicRanking(OnlineRatingSystem):
    def __init__(
        self,
        num_competitors: int,
        initial_mu: float = 0.0,
        initial_sigma: float = 1.2,
        sigma_reduction_factor: float = 1 / 5,  # A in the paper
        sigma_lower_bound: float = 0.4,  # B in the paper
        b: float = 1.0,
        update_method: str = 'iterative',
        dtype=np.float64,
    ):
        self.num_competitors = num_competitors
        self.pi = np.ones(shape=num_competitors, dtype=dtype) / num_competitors
        self.P = np.ones(shape=(num_competitors, num_competitors), dtype=dtype) / (2.0 * num_competitors)
        self.A = np.zeros(shape=(num_competitors, num_competitors), dtype=dtype)

        self.cache = {'probs': None}
        if update_method == 'batched':
            self.update_fn = self.batched_update
        elif update_method == 'iterative':
            self.update_fn = self.iterative_update

    def predict(self, time_step: int, matchups: np.ndarray, set_cache: bool = False):
        """generate predictions"""
        pis_1 = self.mus[matchups[:, 0]]
        pis_2 = self.mus[matchups[:, 1]]
        probs = pis_1 / (pis_1 + pis_2)
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

    def rank_one_update(self, idx, delta):
        pass

    def batched_update(self, matchups, outcomes, use_cache):
        """apply one update based on all of the results of the rating period"""
        raise NotImplementedError

    def iterative_update(self, matchups, outcomes, **kwargs):
        """treat the matchups in the rating period as if they were sequential"""
        for idx in range(matchups.shape[0]):
            pass
            # comp_1, comp_2 = matchups[idx]
            # outcome = outcomes[idx]
            # mu_1, mu_2 = self.mus[matchups[idx]]

    def print_top_k(self, k, competitor_names):
        sorted_idxs = np.argsort(-(self.mus - 3.0 * np.sqrt(self.vs)))[:k]
        for k_idx in range(k):
            comp_idx = sorted_idxs[k_idx]
            print(competitor_names[comp_idx], self.mus[comp_idx])
