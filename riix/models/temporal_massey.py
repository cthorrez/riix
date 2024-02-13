"""
Temporalized Massey Method
https://www.degruyter.com/document/doi/10.1515/jqas-2016-0093
https://arxiv.org/abs/1702.00585

"""
import numpy as np
from riix.core.base import OnlineRatingSystem
from riix.models.elo import Elo


class TemporalMassey(Elo):
    rating_dim = 1

    def __init__(
        self,
        competitors: list,
        initial_rating: float = 1.0,
        alpha: float = 0.95,
        beta: float = 0.05,
        update_method: str = 'iterative',
        dtype=np.float64,
    ):
        OnlineRatingSystem.__init__(self, competitors)
        self.ratings = np.zeros(shape=self.num_competitors, dtype=dtype) + initial_rating
        if (alpha is not None) and (beta is not None):
            self.alpha = alpha
            self.beta = beta
            self.get_coefs = self.get_constant_coefs
        else:
            self.get_coefs = self.get_varying_coefs
            self.matchup_counts = np.zeros(shape=self.num_competitors, dtype=dtype)

        self.cache = {'probs': None}
        if update_method == 'batched':
            self.update = self.batched_update
        elif update_method == 'iterative':
            self.update = self.iterative_update

    def get_pre_match_ratings(self, matchups: np.ndarray, **kwargs):
        return self.ratings[matchups]

    def get_constant_coefs(self, **kwargs):
        return self.alpha, self.beta, self.alpha, self.beta

    def get_varying_coefs_helper(self, idx):
        count = self.matchup_counts[idx]
        if count == 0:
            alpha, beta = 0.0, 0.0
        else:
            alpha = (count - 1.0) / count
            beta = 1.0 / count
        self.matchup_counts[idx] += 1
        return alpha, beta

    def get_varying_coefs(self, idx_1, idx_2):
        alpha_1, beta_1 = self.get_varying_coefs_helper(idx_1)
        alpha_2, beta_2 = self.get_varying_coefs_helper(idx_2)
        return alpha_1, beta_1, alpha_2, beta_2

    def predict(self, matchups: np.ndarray, set_cache: bool = False, **kwargs):
        """generate predictions"""
        ratings_1 = self.ratings[matchups[:, 0]]
        ratings_2 = self.ratings[matchups[:, 1]]
        probs = ratings_1 / (ratings_1 + ratings_2)
        if set_cache:
            self.cache['probs'] = probs
        return probs

    def batched_update(self, matchups, outcomes, use_cache, **kwargs):
        """apply one update based on all of the results of the rating period"""
        pass

    def iterative_update(self, matchups, outcomes, **kwargs):
        """treat the matchups in the rating period as if they were sequential"""
        for idx in range(matchups.shape[0]):
            comp_1, comp_2 = matchups[idx]
            r_1 = self.ratings[comp_1]
            r_2 = self.ratings[comp_2]
            a_1, b_1, a_2, b_2 = self.get_coefs(idx_1=comp_1, idx_2=comp_2)
            s_1 = (outcomes[idx] - 0.5) * 2.0
            s_2 = -1.0 * s_1
            r_1_new = (a_1 * r_1) + (b_1 * r_2) + (b_1 * s_1)
            r_2_new = (a_2 * r_2) + (b_2 * r_1) + (b_2 * s_2)
            self.ratings[comp_1] = r_1_new
            self.ratings[comp_2] = r_2_new
