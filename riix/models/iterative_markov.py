"""
Iterative Markov Method
https://www.degruyter.com/document/doi/10.1515/jqas-2019-0070/html
"""
import numpy as np
from riix.core.base import OnlineRatingSystem


class IterativeMarkov(OnlineRatingSystem):
    def __init__(
        self,
        num_competitors: int,
        initial_rating: float = 1.0,
        c: float = 0.1,
        weight_with_prob: bool = False,  # if True this becomes "Linear Elo"
        update_method: str = 'iterative',
        dtype=np.float64,
    ):
        self.num_competitors = num_competitors
        self.c = c
        self.weight_with_prob = weight_with_prob
        self.ratings = np.zeros(shape=num_competitors, dtype=dtype) + initial_rating
        self.cache = {'probs': None}
        if update_method == 'batched':
            self.update_fn = self.batched_update
        elif update_method == 'iterative':
            self.update_fn = self.iterative_update

    def predict(self, time_step: int, matchups: np.ndarray, set_cache: bool = False):
        """generate predictions"""
        ratings_1 = self.ratings[matchups[:, 0]]
        ratings_2 = self.ratings[matchups[:, 1]]
        probs = ratings_1 / (ratings_1 + ratings_2)
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
        pass

    def iterative_update(self, matchups, outcomes, **kwargs):
        """treat the matchups in the rating period as if they were sequential"""
        for idx in range(matchups.shape[0]):
            comp_1, comp_2 = matchups[idx]
            r_1 = self.ratings[comp_1]
            r_2 = self.ratings[comp_2]
            r_sum = r_1 + r_2
            prob_1 = r_1 / r_sum
            update = self.c * r_sum * (outcomes[idx] - prob_1)
            if self.weight_with_prob:
                update *= prob_1
            self.ratings[comp_1] += update
            self.ratings[comp_2] -= update

    def print_top_k(self, k, competitor_names):
        sorted_idxs = np.argsort(-self.ratings)[:k]
        for k_idx in range(k):
            comp_idx = sorted_idxs[k_idx]
            print(competitor_names[comp_idx], self.ratings[comp_idx])
