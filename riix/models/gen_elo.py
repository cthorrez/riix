"""Bayesian generalization of Elo by Martin Ingram
https://www.researchgate.net/publication/348511584_How_to_extend_Elo_a_Bayesian_perspective
equations 17-19 are relevant for the case without adding surface, and best-of effects"""

import math
import numpy as np
import jax
import jax.numpy as jnp

from riix.core.base import OnlineRatingSystem


class GenElo(OnlineRatingSystem):
    """Bayesian generalized Elo."""

    rating_dim = 1

    def __init__(
        self,
        competitors: list,
        initial_mu: float = 0.0,
        b: float = math.log(10.0) / 400.0,
        sigma: float = 60.0,
        use_approx: bool = True,
        update_method: str = 'online',
        predict_mode: str = 'prequential',
        dtype=jnp.float32,
    ):
        """
        Args:
            initial_mu (float, optional): initial skill mean for every competitor. Defaults to 0.0.
            b (float, optional): rating-difference scale inside the sigmoid. Defaults to log(10)/400.
            sigma (float, optional): performance noise standard deviation. Defaults to 60.0.
            use_approx (bool, optional): apply Ingram's logistic-normal approximation factor 1/sqrt(1 + pi * sigma^2 * b^2 / 8) to the update. Defaults to True.
            update_method / predict_mode / dtype: see `OnlineRatingSystem.__init__`.
        """
        super().__init__(competitors, dtype=dtype, update_method=update_method, predict_mode=predict_mode)
        self.initial_mu = initial_mu
        sigma_delta_squared = 2.0 * sigma**2.0
        alpha = math.sqrt(1.0 + ((math.pi * sigma_delta_squared * b**2.0) / 8.0)) if use_approx else 1.0
        self.hparams = {
            'b': jnp.asarray(b, dtype=self.dtype),
            'sigma_2_delta_inv': jnp.asarray(1.0 / sigma_delta_squared, dtype=self.dtype),
            'alpha': jnp.asarray(alpha, dtype=self.dtype),
        }

    def _initial_state(self):
        return (jnp.full(self.num_competitors, self.initial_mu, dtype=self.dtype),)

    @staticmethod
    def _step_core(state, x, hparams):
        (mus,) = state
        matchup, outcome = x
        mu = mus[matchup]
        b = hparams['b']
        diff = mu[0] - mu[1]
        # v1 quirk preserved: predictions temper the sigmoid by alpha, while the
        # update's internal probability does not
        prob = jax.nn.sigmoid(b * diff / hparams['alpha'])
        update_prob = jax.nn.sigmoid(b * diff)
        k = (b / 2.0) / (hparams['sigma_2_delta_inv'] + (b * b * update_prob * (1.0 - update_prob)))
        delta = k * (outcome - update_prob)
        mus = mus.at[matchup].add(jnp.stack([delta, -delta]))
        return (mus,), prob, mu

    @staticmethod
    def _step(state, x, hparams):
        new_state, prob, _ = GenElo._step_core(state, x, hparams)
        return new_state, prob

    @staticmethod
    def _step_period_start(state, x, hparams):
        new_state, _, mu = GenElo._step_core(state, x, hparams)
        return new_state, mu

    def _gathered_probs(self, gathered):
        (mu,) = gathered  # (n, 2) pre-match ratings
        return jax.nn.sigmoid(self.hparams['b'] * (mu[:, 0] - mu[:, 1]) / self.hparams['alpha'])

    @property
    def mus(self):
        return np.asarray(self.state[0])

    def leaderboard_ratings(self):
        return self.mus
