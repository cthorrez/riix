"""
A general class implementing online gradient based rating systems on differentiable likelihoods
"""
import math
import numpy as np
from riix.core.base import OnlineRatingSystem
import jax.numpy as jnp
from jax import grad, jit
import jax.nn
from jax.scipy.stats import logistic


class AutogradRatingSystem(OnlineRatingSystem):
    """update the ratings using gradients"""

    rating_dim = 1

    def __init__(
        self,
        competitors: list,
        cdf: callable = logistic.cdf,
        scale: float = 400.0 / math.log(10.0),
        learning_rate: float = 32.0,
        initial_rating: float = 1500.0,
        update_method: str = 'iterative',
        dtype=jnp.float32,
    ):
        """Initialize the rating system"""
        super().__init__(competitors)

        self.ratings = jnp.zeros(shape=self.num_competitors, dtype=dtype) + initial_rating
        self.cdf = cdf
        self.scale = scale

        @jit
        def likelihood_fn(ratings, matchups, outcomes):
            matchup_ratings = ratings[matchups]
            rating_diffs = matchup_ratings[:, 0] - matchup_ratings[:, 1]
            probs = self.cdf(rating_diffs / self.scale)
            return -jnp.log((outcomes * probs) + ((1.0 - outcomes) * (1.0 - probs))).sum()
        
        @jit
        def _predict_fn(ratings, matchups):
            matchup_ratings = ratings[matchups]
            rating_diffs = matchup_ratings[:, 0] - matchup_ratings[:, 1]
            probs = self.cdf(rating_diffs / self.scale)
            return probs
        self.predict_fn = _predict_fn
        

        self.grad_fn = grad(likelihood_fn)
        self.learning_rate = learning_rate * scale

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
        grad = self.grad_fn(self.ratings, jnp.array(matchups), jnp.array(outcomes))
        update = grad * self.learning_rate
        self.ratings -= update

    def iterative_update(self, matchups, outcomes, **kwargs):
        """
        Treats the matchups in the rating period as sequential events.

        Parameters:
            matchups: Sequential matchups in the rating period.
            outcomes: Results of each matchup.
        """
        for idx in range(matchups.shape[0]):
            matchup = matchups[idx : idx + 1, :]
            outcome = outcomes[idx : idx + 1]
            grad = self.grad_fn(self.ratings, jnp.array(matchup), jnp.array(outcome))
            update = grad * self.learning_rate
            self.ratings -= update

    def print_leaderboard(self, num_places):
        sorted_idxs = np.argsort(-self.ratings)[:num_places]
        max_len = min(np.max([len(comp) for comp in self.competitors] + [10]), 25)
        print(f'{"competitor": <{max_len}}\t{"rating"}')
        for p_idx in range(num_places):
            comp_idx = sorted_idxs[p_idx]
            print(f'{self.competitors[comp_idx]: <{max_len}}\t{self.ratings[comp_idx]:.6f}')
