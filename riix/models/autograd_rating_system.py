"""
A general class implementing online gradient based rating systems on differentiable likelihoods
"""

import math
import numpy as np
from riix.core.base import OnlineRatingSystem

try:
    import jax.numpy as jnp
    from jax import grad, jit, lax
    from jax.scipy.stats import logistic
except ImportError as e:
    raise ImportError(
        'AutogradRatingSystem requires jax, which is an optional dependency. Install it with: uv add "riix[jax]"'
    ) from e


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
        update_method: str = 'online',
        dtype=jnp.float32,
    ):
        """Initialize the rating system"""
        super().__init__(competitors)

        self.ratings = jnp.zeros(shape=self.num_competitors, dtype=dtype) + initial_rating
        self.cdf = cdf
        self.scale = scale
        self.learning_rate = learning_rate * scale
        lr = self.learning_rate

        def likelihood_fn(matchup_ratings, outcomes):
            # differentiating w.r.t. only the gathered ratings keeps the gradient
            # O(matchups) instead of O(num_competitors)
            rating_diffs = matchup_ratings[..., 0] - matchup_ratings[..., 1]
            probs = self.cdf(rating_diffs / self.scale)
            return -jnp.log((outcomes * probs) + ((1.0 - outcomes) * (1.0 - probs))).sum()

        grad_fn = grad(likelihood_fn)

        @jit
        def _batched_update_fn(ratings, matchups, outcomes):
            update = grad_fn(ratings[matchups], outcomes) * lr
            return ratings.at[matchups].add(-update)

        @jit
        def _online_update_fn(ratings, matchups, outcomes):
            def step(ratings, matchup_outcome):
                matchup, outcome = matchup_outcome
                update = grad_fn(ratings[matchup], outcome) * lr
                return ratings.at[matchup].add(-update), None

            ratings, _ = lax.scan(step, ratings, (matchups, outcomes))
            return ratings

        @jit
        def _predict_fn(ratings, matchups):
            matchup_ratings = ratings[matchups]
            rating_diffs = matchup_ratings[:, 0] - matchup_ratings[:, 1]
            probs = self.cdf(rating_diffs / self.scale)
            return probs

        self.predict_fn = _predict_fn
        self.batched_update_fn = _batched_update_fn
        self.online_update_fn = _online_update_fn

        if update_method == 'batched':
            self.update = self.batched_update
        elif update_method == 'online':
            self.update = self.online_update
        else:
            raise ValueError(f'update_method must be one of online or batched, got {update_method}')

    def _pad(self, matchups: np.ndarray, outcomes: np.ndarray = None):
        """
        Pad a batch to the next power-of-two length so jit compiles O(log(max batch size))
        shapes instead of one per distinct rating period size. Padding rows are self-matchups
        with outcome 0.5, whose likelihood gradient is exactly zero for any cdf.
        """
        num_matchups = matchups.shape[0]
        padded_len = 1 << (num_matchups - 1).bit_length()
        if padded_len != num_matchups:
            matchups = np.concatenate([matchups, np.zeros((padded_len - num_matchups, 2), dtype=matchups.dtype)])
            if outcomes is not None:
                outcomes = np.concatenate([outcomes, np.full(padded_len - num_matchups, 0.5, dtype=outcomes.dtype)])
        if outcomes is None:
            return jnp.array(matchups)
        return jnp.array(matchups), jnp.array(outcomes)

    def predict(self, matchups: np.ndarray, time_step: int = None, set_cache: bool = False):
        """
        Generates predictions for a series of matchups between competitors.
        """
        return self.predict_fn(self.ratings, self._pad(matchups))[: matchups.shape[0]]

    def get_pre_match_ratings(self, matchups: np.ndarray, **kwargs):
        return self.ratings[matchups]

    def batched_update(self, matchups, outcomes, use_cache, **kwargs):
        """
        Apply a single update based on all results of the rating period.

        Parameters:
            matchups: Matchup information for the rating period.
            outcomes: Results of the matchups.
        """
        matchups, outcomes = self._pad(matchups, outcomes)
        self.ratings = self.batched_update_fn(self.ratings, matchups, outcomes)

    def online_update(self, matchups, outcomes, **kwargs):
        """
        Treats the matchups in the rating period as sequential events.

        Parameters:
            matchups: Sequential matchups in the rating period.
            outcomes: Results of each matchup.
        """
        matchups, outcomes = self._pad(matchups, outcomes)
        self.ratings = self.online_update_fn(self.ratings, matchups, outcomes)

    def print_leaderboard(self, num_places):
        sorted_idxs = np.argsort(-self.ratings)[:num_places]
        max_len = min(np.max([len(comp) for comp in self.competitors] + [10]), 25)
        print(f'{"competitor": <{max_len}}\t{"rating"}')
        for p_idx in range(num_places):
            comp_idx = sorted_idxs[p_idx]
            print(f'{self.competitors[comp_idx]: <{max_len}}\t{self.ratings[comp_idx]:.6f}')
