"""variance incorporated Elo
Rating of players by Laplace approximation and dynamic modeling
https://arxiv.org/abs/2310.10386"""

import numpy as np
import jax
import jax.numpy as jnp

from riix.core.base import OnlineRatingSystem


class vElo(OnlineRatingSystem):
    """Variance incorporated Elo.

    Carries (mus, variances); variances shrink per match by the Laplace-approximation
    reduction factor and are floored at the lower bound. No time dynamics.
    """

    rating_dim = 2
    leaderboard_header = 'mu - (3 * sd)'
    _predict_state_slots = (0, 1)  # (mus, variances)

    def __init__(
        self,
        competitors: list,
        initial_mu: float = 0.0,
        initial_sigma: float = 1.2,
        sigma_reduction_factor: float = 1 / 5,  # A in the paper
        sigma_lower_bound: float = 0.4,  # B in the paper
        b: float = 1.0,
        update_method: str = 'online',
        predict_mode: str = 'prequential',
        dtype=jnp.float32,
    ):
        """
        Args:
            initial_mu (float, optional): initial skill mean for every competitor. Defaults to 0.0.
            initial_sigma (float, optional): initial skill standard deviation. Defaults to 1.2.
            sigma_reduction_factor (float, optional): multiplicative variance shrink on surprising results (the paper's A). Defaults to 1/5.
            sigma_lower_bound (float, optional): floor on the standard deviation (the paper's B). Defaults to 0.4.
            b (float, optional): rating-difference scale inside the sigmoid. Defaults to 1.0.
            update_method / predict_mode / dtype: see `OnlineRatingSystem.__init__`.
        """
        super().__init__(competitors, dtype=dtype, update_method=update_method, predict_mode=predict_mode)
        self.initial_mu = initial_mu
        self.initial_v = initial_sigma**2.0
        self.hparams = {
            'b': jnp.asarray(b, dtype=self.dtype),
            'reduction_factor': jnp.asarray(sigma_reduction_factor, dtype=self.dtype),
            'v_lower_bound': jnp.asarray(sigma_lower_bound**2.0, dtype=self.dtype),
        }

    def _initial_state(self):
        return (
            jnp.full(self.num_competitors, self.initial_mu, dtype=self.dtype),
            jnp.full(self.num_competitors, self.initial_v, dtype=self.dtype),
        )

    @staticmethod
    def _step_core(state, x, hparams):
        mus, vs = state
        matchup, outcome = x
        mu = mus[matchup]
        v = vs[matchup]
        b = hparams['b']
        b2 = b * b
        prob = jax.nn.sigmoid(b * (mu[0] - mu[1]))
        delta = outcome - prob
        v_sum = v[0] + v[1]

        c = 1.0 / (1.0 + (b2 * prob * (1.0 - prob) * v_sum))
        mu_new = mu + (v * b * c * delta * jnp.array([1.0, -1.0], dtype=mu.dtype))

        prob_p = jax.nn.sigmoid(b * (mu_new[0] - mu_new[1]))
        prob_prob = prob_p * (1.0 - prob_p)
        c_p = 1.0 / (1.0 + (b2 * prob_prob * v_sum))
        ell = prob_prob * v * b2 * c_p
        v_new = jnp.maximum(hparams['v_lower_bound'], v * (1.0 - (hparams['reduction_factor'] * ell)))

        mus = mus.at[matchup].set(mu_new)
        vs = vs.at[matchup].set(v_new)
        return (mus, vs), prob, mu, v

    @staticmethod
    def _step(state, x, hparams):
        new_state, prob, _, _ = vElo._step_core(state, x, hparams)
        return new_state, prob

    @staticmethod
    def _step_period_start(state, x, hparams):
        new_state, _, mu, v = vElo._step_core(state, x, hparams)
        return new_state, (mu, v)

    def _gathered_probs(self, gathered):
        mu = gathered[0]  # (n, 2) pre-match means; the variances don't enter the prediction
        return jax.nn.sigmoid(self.hparams['b'] * (mu[:, 0] - mu[:, 1]))

    @property
    def mus(self):
        return np.asarray(self.state[0])

    @property
    def vs(self):
        return np.asarray(self.state[1])

    def leaderboard_ratings(self):
        return self.mus - (3.0 * np.sqrt(self.vs))
