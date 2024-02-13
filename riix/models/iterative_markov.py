"""
Iterative Markov Method
https://www.degruyter.com/document/doi/10.1515/jqas-2019-0070
"""
import numpy as np
from riix.core.base import OnlineRatingSystem
from riix.models.elo import Elo


class IterativeMarkov(Elo):
    """Iterative Markov rating system"""

    rating_dim = 1

    def __init__(
        self,
        competitors: list,
        initial_rating: float = 1.0,
        c: float = 0.1,
        weight_with_prob: bool = False,  # if True this becomes "Linear Elo"
        update_method: str = 'iterative',
        dtype=np.float64,
    ):
        OnlineRatingSystem.__init__(self, competitors)
        self.c = c
        self.weight_with_prob = weight_with_prob
        self.ratings = np.zeros(shape=self.num_competitors, dtype=dtype) + initial_rating
        self.cache = {'probs': None}
        if update_method == 'batched':
            self.update = self.batched_update
        elif update_method == 'iterative':
            self.update = self.iterative_update

    def get_pre_match_ratings(self, matchups: np.ndarray, **kwargs):
        return self.ratings[matchups]

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

    def iterative_update(self, matchups, outcomes, use_cache=False, **kwargs):
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
