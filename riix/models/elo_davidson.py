"""The Elo-Davidson rating system"""

import math
import numpy as np
import jax
import jax.numpy as jnp

from riix.core.base import OnlineRatingSystem


class EloDavidson(OnlineRatingSystem):
    """Elo-Davidson."""

    rating_dim = 1

    def __init__(
        self,
        competitors: list,
        initial_rating: float = 0.0,
        k: float = 32.0,
        kappa: float = 1.0,
        base: float = 10.0,
        sigma: float = 200.0,
        update_method: str = 'online',
        predict_mode: str = 'prequential',
        dtype=jnp.float32,
    ):
        """
        Args:
            initial_rating (float, optional): The initial Elo rating for new competitors. Defaults to 1500.0.
            k (float, optional): update step size (K-factor). Defaults to 32.0.
            kappa (float, optional): Davidson draw parameter — the mass placed on draws in the expected score (0 recovers plain Elo). Defaults to 1.0.
            base (float, optional): exponent base of the rating-difference scale. Defaults to 10.0.
            sigma (float, optional): rating-difference scale divisor. Defaults to 200.0.
            update_method / predict_mode / dtype: see `OnlineRatingSystem.__init__`.
        """
        super().__init__(competitors, dtype=dtype, update_method=update_method, predict_mode=predict_mode)
        self.initial_rating = initial_rating
        self.hparams = {
            'k': jnp.asarray(k, dtype=self.dtype),
            'kappa': jnp.asarray(kappa, dtype=self.dtype),
            'alpha': jnp.asarray(math.log(base) / (2.0 * sigma), dtype=self.dtype),
        }

    def _initial_state(self):
        return (jnp.full(self.num_competitors, self.initial_rating, dtype=self.dtype),)

    @staticmethod
    def _step_core(state, x, hparams):
        (ratings,) = state
        matchup, outcome = x
        r = ratings[matchup]
        # v1 quirk preserved: predictions are plain Elo sigmoids, while updates
        # move ratings toward the Davidson expected score (draws get kappa mass)
        prob = jax.nn.sigmoid(hparams['alpha'] * (r[0] - r[1]))
        diff = jnp.exp(hparams['alpha'] * (r[0] - r[1]))
        kappa_over_2 = hparams['kappa'] / 2.0
        denom = diff + (1.0 / diff) + hparams['kappa']
        p_1 = (diff + kappa_over_2) / denom
        p_2 = ((1.0 / diff) + kappa_over_2) / denom
        update_1 = hparams['k'] * (outcome - p_1)
        update_2 = hparams['k'] * (1.0 - outcome - p_2)
        ratings = ratings.at[matchup].add(jnp.stack([update_1, update_2]))
        return (ratings,), prob, r

    @staticmethod
    def _step(state, x, hparams):
        new_state, prob, _ = EloDavidson._step_core(state, x, hparams)
        return new_state, prob

    @staticmethod
    def _step_period_start(state, x, hparams):
        new_state, _, r = EloDavidson._step_core(state, x, hparams)
        return new_state, r

    def _gathered_probs(self, gathered):
        (r,) = gathered  # (n, 2) pre-match ratings
        return jax.nn.sigmoid(self.hparams['alpha'] * (r[:, 0] - r[:, 1]))

    @property
    def ratings(self):
        return np.asarray(self.state[0])

    def leaderboard_ratings(self):
        return self.ratings
