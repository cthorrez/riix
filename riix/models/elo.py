"""The Elo rating system"""

import math
import numpy as np
import jax
import jax.numpy as jnp

from riix.core.base import OnlineRatingSystem


class Elo(OnlineRatingSystem):
    """The original Elo rating system."""

    rating_dim = 1

    def __init__(
        self,
        competitors: list,
        initial_rating: float = 1500.0,
        k: float = 32.0,
        alpha: float = math.log(10.0) / 400.0,
        update_method: str = 'online',
        predict_mode: str = 'prequential',
        dtype=jnp.float32,
    ):
        """
        Args:
            initial_rating (float, optional): The initial Elo rating for new competitors. Defaults to 1500.0.
            k (float, optional): The K-factor, which controls the rate at which ratings change. Defaults to 32.0.
            alpha (float, optional): Scaling factor used in the calculation of expected scores. Defaults to log(10) / 400.
            update_method / predict_mode / dtype: see `OnlineRatingSystem.__init__`.
        """
        super().__init__(competitors, dtype=dtype, update_method=update_method, predict_mode=predict_mode)
        self.initial_rating = initial_rating
        self.hparams = {
            'k': jnp.asarray(k, dtype=self.dtype),
            'alpha': jnp.asarray(alpha, dtype=self.dtype),
        }

    def _initial_state(self):
        return (jnp.full(self.num_competitors, self.initial_rating, dtype=self.dtype),)

    @staticmethod
    def _step_core(state, x, hparams):
        (ratings,) = state
        matchup, outcome = x
        r = ratings[matchup]
        prob = jax.nn.sigmoid(hparams['alpha'] * (r[0] - r[1]))
        update = hparams['k'] * (outcome - prob)
        ratings = ratings.at[matchup].add(jnp.stack([update, -update]))
        return (ratings,), prob, r

    @staticmethod
    def _step(state, x, hparams):
        new_state, prob, _ = Elo._step_core(state, x, hparams)
        return new_state, prob

    @staticmethod
    def _step_period_start(state, x, hparams):
        new_state, _, r = Elo._step_core(state, x, hparams)
        return new_state, r

    def _gathered_probs(self, gathered):
        (r,) = gathered  # (n, 2) pre-match ratings
        return jax.nn.sigmoid(self.hparams['alpha'] * (r[:, 0] - r[:, 1]))

    @staticmethod
    def _batched_step(state, x, hparams):
        (ratings,) = state
        matchups, outcomes, mask, _time_delta = x
        r = ratings[matchups]  # (p, 2)
        probs = jax.nn.sigmoid(hparams['alpha'] * (r[:, 0] - r[:, 1]))
        diff = (outcomes - probs) * mask
        updates = hparams['k'] * jnp.stack([diff, -diff], axis=1)
        ratings = ratings.at[matchups.reshape(-1)].add(updates.reshape(-1))
        return (ratings,), probs

    @property
    def ratings(self):
        state = self.state if self.state is not None else self._initial_state()
        return np.asarray(state[0])

    @ratings.setter
    def ratings(self, values):
        self._set_state_slot(0, values)

    def leaderboard_ratings(self):
        return self.ratings
