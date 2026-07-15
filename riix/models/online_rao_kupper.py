"""The Online Rao Kupper rating system"""

import math
import numpy as np
import jax
import jax.numpy as jnp

from riix.core.base import OnlineRatingSystem


class OnlineRaoKupper(OnlineRatingSystem):
    """Online Rao-Kupper."""

    rating_dim = 1

    def __init__(
        self,
        competitors: list,
        initial_rating: float = 0.0,
        theta: float = 2.0,
        step_size: float = 32.0,
        temperature: float = math.log(10.0) / 400.0,
        update_method: str = 'online',
        predict_mode: str = 'prequential',
        dtype=jnp.float32,
    ):
        """
        Args:
            initial_rating (float, optional): The initial Elo rating for new competitors. Defaults to 1500.0.
            theta (float, optional): Rao-Kupper draw threshold (1 makes draws impossible). Defaults to 2.0.
            step_size (float, optional): gradient-ascent step size. Defaults to 32.0.
            temperature (float, optional): rating-difference scale inside the model. Defaults to log(10)/400.
            update_method / predict_mode / dtype: see `OnlineRatingSystem.__init__`.
        """
        super().__init__(competitors, dtype=dtype, update_method=update_method, predict_mode=predict_mode)
        self.initial_rating = initial_rating
        self.hparams = {
            'log_theta': jnp.asarray(math.log(theta), dtype=self.dtype),
            'step_size': jnp.asarray(step_size, dtype=self.dtype),
            'temperature': jnp.asarray(temperature, dtype=self.dtype),
        }

    def _initial_state(self):
        return (jnp.full(self.num_competitors, self.initial_rating, dtype=self.dtype),)

    @staticmethod
    def _step_core(state, x, hparams):
        (ratings,) = state
        matchup, outcome = x
        r = ratings[matchup]
        temp = hparams['temperature']
        diff = r[0] - r[1]
        # v1 quirk preserved: predictions ignore theta, gradients include it
        prob = jax.nn.sigmoid(temp * diff)
        p_1 = jax.nn.sigmoid(temp * (diff - hparams['log_theta']))
        p_2 = jax.nn.sigmoid(temp * (-diff - hparams['log_theta']))
        is_draw = outcome == 0.5
        g_1 = jnp.where(is_draw, p_2 - p_1, outcome - p_1)
        g_2 = jnp.where(is_draw, p_1 - p_2, (1.0 - outcome) - p_2)
        ratings = ratings.at[matchup].add(hparams['step_size'] * jnp.stack([g_1, g_2]))
        return (ratings,), prob, r

    @staticmethod
    def _step(state, x, hparams):
        new_state, prob, _ = OnlineRaoKupper._step_core(state, x, hparams)
        return new_state, prob

    @staticmethod
    def _step_period_start(state, x, hparams):
        new_state, _, r = OnlineRaoKupper._step_core(state, x, hparams)
        return new_state, r

    def _gathered_probs(self, gathered):
        (r,) = gathered  # (n, 2) pre-match ratings
        return jax.nn.sigmoid(self.hparams['temperature'] * (r[:, 0] - r[:, 1]))

    @property
    def ratings(self):
        return np.asarray(self.state[0])

    def leaderboard_ratings(self):
        return self.ratings
