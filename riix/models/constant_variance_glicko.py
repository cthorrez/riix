"""Constant Variance Glicko
Martin Ingram 2021
https://www.researchgate.net/publication/348511584_How_to_extend_Elo_a_Bayesian_perspective"""

import math
import numpy as np
import jax
import jax.numpy as jnp

from riix.core.base import OnlineRatingSystem


class ConstantVarianceGlicko(OnlineRatingSystem):
    """Constant Variance Glicko.

    All competitors share an equal, unchanging rating deviation, so unlike
    Glicko there is no variance state and no time-based inflation pre-pass.
    """

    rating_dim = 1

    def __init__(
        self,
        competitors: list,
        initial_rating: float = 1500.0,
        rd: float = 81.7,  # for tennis Ingram found this should be 81.7
        b: float = math.log(10.0) / 400.0,
        update_method: str = 'online',
        predict_mode: str = 'prequential',
        dtype=jnp.float32,
    ):
        """
        Args:
            initial_rating (float, optional): initial rating for every competitor. Defaults to 1500.0.
            rd (float, optional): the shared, constant rating deviation used for every competitor (81.7 per Ingram for tennis). Defaults to 81.7.
            b (float, optional): rating-difference scale inside the sigmoid. Defaults to log(10)/400.
            update_method / predict_mode / dtype: see `OnlineRatingSystem.__init__`.
        """
        super().__init__(competitors, dtype=dtype, update_method=update_method, predict_mode=predict_mode)
        self.initial_rating = initial_rating
        g_rd = 1.0 / math.sqrt(1.0 + (3.0 * (b**2.0) * (rd**2.0)) / (math.pi**2.0))
        self.hparams = {
            'rd2_inv': jnp.asarray(1.0 / (rd**2.0), dtype=self.dtype),
            'b_g_rd': jnp.asarray(b * g_rd, dtype=self.dtype),
        }

    def _initial_state(self):
        return (jnp.full(self.num_competitors, self.initial_rating, dtype=self.dtype),)

    @staticmethod
    def _step_core(state, x, hparams):
        (ratings,) = state
        matchup, outcome = x
        r = ratings[matchup]
        b_g_rd = hparams['b_g_rd']
        prob = jax.nn.sigmoid(b_g_rd * (r[0] - r[1]))
        d_squared_inv = b_g_rd * b_g_rd * prob * (1.0 - prob)
        k = b_g_rd / (hparams['rd2_inv'] + d_squared_inv)
        delta = k * (outcome - prob)
        ratings = ratings.at[matchup].add(jnp.stack([delta, -delta]))
        return (ratings,), prob, r

    @staticmethod
    def _step(state, x, hparams):
        new_state, prob, _ = ConstantVarianceGlicko._step_core(state, x, hparams)
        return new_state, prob

    @staticmethod
    def _step_period_start(state, x, hparams):
        new_state, _, r = ConstantVarianceGlicko._step_core(state, x, hparams)
        return new_state, r

    def _gathered_probs(self, gathered):
        (r,) = gathered  # (n, 2) pre-match ratings
        return jax.nn.sigmoid(self.hparams['b_g_rd'] * (r[:, 0] - r[:, 1]))

    @property
    def ratings(self):
        return np.asarray(self.state[0])

    def leaderboard_ratings(self):
        return self.ratings
