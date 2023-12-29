"""
variance incorporated Elo
Rating of players by Laplace approximation and dynamic modeling
https://arxiv.org/abs/2310.10386
"""
import numpy as np
from riix.core.base import OnlineRatingSystem
from riix.utils.math_utils import sigmoid


class vElo(OnlineRatingSystem):
    def __init__(
        self,
        num_competitors: int,
        # initial_mu: float = 1500.0,
        # initial_sigma: float = 250.0,
        # sigma_reduction_factor: float = 1 / 5,  # A in the paper
        # sigma_lower_bound: float = 100.0,  # B in the paper
        # b: float = math.log(10.0) / 400.0,
        initial_mu: float = 0.0,
        initial_sigma: float = 1.2,
        sigma_reduction_factor: float = 1 / 5,  # A in the paper
        sigma_lower_bound: float = 0.4,  # B in the paper
        b: float = 1.0,
        update_method: str = 'iterative',
        dtype=np.float64,
    ):
        self.num_competitors = num_competitors
        self.mus = np.zeros(num_competitors, dtype=dtype) + initial_mu
        self.vs = np.zeros(num_competitors, dtype=dtype) + np.square(initial_sigma)
        self.sigma_reduction_factor = sigma_reduction_factor
        self.variance_lower_bound = sigma_lower_bound**2.0
        self.b = b
        self.b2 = b**2.0
        self.cache = {'probs': None}
        if update_method == 'batched':
            self.update_fn = self.batched_update
        elif update_method == 'iterative':
            self.update_fn = self.iterative_update

    def predict(self, time_step: int, matchups: np.ndarray, set_cache: bool = False):
        """generate predictions"""
        mus_1 = self.mus[matchups[:, 0]]
        mus_2 = self.mus[matchups[:, 1]]
        probs = sigmoid(self.b * (mus_1 - mus_2))
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
        raise NotImplementedError

    def iterative_update(self, matchups, outcomes, **kwargs):
        """treat the matchups in the rating period as if they were sequential"""
        for idx in range(matchups.shape[0]):
            comp_1, comp_2 = matchups[idx]
            outcome = outcomes[idx]
            mu_1, mu_2 = self.mus[matchups[idx]]
            v_1, v_2 = self.vs[matchups[idx]]
            prob = sigmoid(self.b * (mu_1 - mu_2))
            delta = outcome - prob

            C = 1.0 / (1.0 + self.b2 * prob * (1.0 - prob) * (v_1 + v_2))
            k_1 = v_1 * self.b * C
            k_2 = v_2 * self.b * C
            mu_1_p = mu_1 + (k_1 * delta)
            mu_2_p = mu_2 - (k_2 * delta)

            prob_p = sigmoid(self.b * (mu_1_p - mu_2_p))
            prob_prob = prob_p * (1.0 - prob_p)
            C_p = 1.0 / (1.0 + self.b2 * prob_prob * (v_1 + v_2))
            L_1 = prob_prob * v_1 * self.b2 * C_p
            L_2 = prob_prob * v_2 * self.b2 * C_p

            v_1_p = max(self.variance_lower_bound, v_1 * (1.0 - (self.sigma_reduction_factor * L_1)))
            v_2_p = max(self.variance_lower_bound, v_2 * (1.0 - (self.sigma_reduction_factor * L_2)))

            self.mus[comp_1] = mu_1_p
            self.mus[comp_2] = mu_2_p
            self.vs[comp_1] = v_1_p
            self.vs[comp_2] = v_2_p

    def print_top_k(self, k, competitor_names):
        sorted_idxs = np.argsort(-(self.mus - 3.0 * np.sqrt(self.vs)))[:k]
        for k_idx in range(k):
            comp_idx = sorted_idxs[k_idx]
            print(competitor_names[comp_idx], self.mus[comp_idx])
