"""
A general class implementing online gradient based rating systems on differentiable likelihoods
"""
import math
import numpy as np
from riix.core.base import OnlineRatingSystem
import jax.numpy as jnp
from jax import grad


def logistic_predict(ratings, matchups, alpha=math.log(10.0) / 400.0):
    matchup_ratings = ratings[matchups]
    neg_rating_diffs = matchup_ratings[:, 1] - matchup_ratings[:, 0]
    probs = 1.0 / (1.0 + jnp.exp(alpha * neg_rating_diffs))
    return probs


def logistic_likelihood(ratings, matchups, outcomes, alpha=math.log(10.0) / 400.0):
    matchup_ratings = ratings[matchups]
    neg_rating_diffs = matchup_ratings[:, 1] - matchup_ratings[:, 0]
    probs = 1.0 / (1.0 + jnp.exp(alpha * neg_rating_diffs))
    return jnp.log(outcomes - probs).sum()


class AutogradRatingSystem(OnlineRatingSystem):
    """update the ratings using gradients"""

    rating_dim = 1

    def __init__(
        self,
        competitors: list,
        predict_fn: callable = logistic_predict,
        likelihood_fn: callable = logistic_likelihood,
        learning_rate: float = 32.0,
        update_method: str = 'iterative',
        dtype=np.float64,
    ):
        """Initialize the rating system"""
        super().__init__(competitors)

        self.ratings = jnp.zeros(shape=self.num_competitors, dtype=dtype)

        self.predict_fn = predict_fn
        self.grad_fn = grad(likelihood_fn)
        self.learning_rate = learning_rate

        if update_method == 'batched':
            self.update = self.batched_update
        elif update_method == 'iterative':
            self.update = self.iterative_update

    def predict(self, matchups: np.ndarray, time_step: int = None, set_cache: bool = False):
        """
        Generates predictions for a series of matchups between competitors.
        """
        return self.predict_fn(self.ratings, matchups)

    def get_pre_match_ratings(self, matchups: np.ndarray, **kwargs):
        return self.ratings[matchups]

    def batched_update(self, matchups, outcomes, use_cache, **kwargs):
        """
        Apply a single update based on all results of the rating period.

        Parameters:
            matchups: Matchup information for the rating period.
            outcomes: Results of the matchups.
        """
        raise NotImplementedError

    def iterative_update(self, matchups, outcomes, **kwargs):
        """
        Treats the matchups in the rating period as sequential events.

        Parameters:
            matchups: Sequential matchups in the rating period.
            outcomes: Results of each matchup.
        """
        for idx in range(matchups.shape[0]):
            matchup = matchups[idx : idx + 1, :]
            outcome = outcomes[idx, idx + 1]
            grad = self.grad_fn(self.ratings, jnp.array(matchup), jnp.array(outcome))
            self.ratings += self.learning_rate * grad

    def print_leaderboard(self, num_places):
        raise NotImplementedError
