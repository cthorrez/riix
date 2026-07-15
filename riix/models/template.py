"""A template to copy and paste when implementing a new rating system.

The whole fit is one jitted `lax.scan` (see `riix.core.base`): implement the
per-match step functions as staticmethods (their identity is the jit key, so
the compiled scan is shared across instances), keep all hyperparameters as
traced jnp scalars in `self.hparams` (sweeps then never recompile), and expose
state through numpy-converting properties.
"""

import numpy as np
import jax
import jax.numpy as jnp

from riix.core.base import OnlineRatingSystem


class Template(OnlineRatingSystem):
    """Put a one-line description of the rating system here."""

    rating_dim = 1

    def __init__(
        self,
        competitors: list,
        initial_rating: float = 1500.0,
        step_size: float = 32.0,
        update_method: str = 'online',
        predict_mode: str = 'prequential',
        dtype=jnp.float32,
    ):
        """
        Args:
            initial_rating (float, optional): initial rating for every competitor. Defaults to 1500.0.
            step_size (float, optional): update step size. Defaults to 32.0.
            update_method / predict_mode / dtype: see `OnlineRatingSystem.__init__`.
        """
        super().__init__(competitors, dtype=dtype, update_method=update_method, predict_mode=predict_mode)
        self.initial_rating = initial_rating
        self.hparams = {
            'step_size': jnp.asarray(step_size, dtype=self.dtype),
        }

    def _initial_state(self):
        """The scan carry: a tuple of jnp arrays."""
        return (jnp.full(self.num_competitors, self.initial_rating, dtype=self.dtype),)

    @staticmethod
    def _step_core(state, x, hparams):
        """Process one match: gather, predict, update, scatter back.

        Returns (new_state, prob, *emitted) where the emitted values are the
        pre-update gathered state — they power period_start predictions,
        return_pre_match_ratings, and the standalone predict().
        """
        (ratings,) = state
        matchup, outcome = x
        r = ratings[matchup]  # (2,) pre-match ratings
        prob = jax.nn.sigmoid(r[0] - r[1])
        update = hparams['step_size'] * (outcome - prob)
        ratings = ratings.at[matchup].add(jnp.stack([update, -update]))
        return (ratings,), prob, r

    @staticmethod
    def _step(state, x, hparams):
        new_state, prob, _ = Template._step_core(state, x, hparams)
        return new_state, prob

    @staticmethod
    def _step_period_start(state, x, hparams):
        new_state, _, r = Template._step_core(state, x, hparams)
        return new_state, r

    def _gathered_probs(self, gathered):
        """The predict() formula, vectorized over (n, 2) gathered pre-match values."""
        (r,) = gathered
        return jax.nn.sigmoid(r[:, 0] - r[:, 1])

    # implement _batched_step (one padded rating period per scan step) only if the
    # system has meaningful simultaneous-update semantics; see Elo for the pattern

    @property
    def ratings(self):
        state = self.state if self.state is not None else self._initial_state()
        return np.asarray(state[0])

    def leaderboard_ratings(self):
        return self.ratings
