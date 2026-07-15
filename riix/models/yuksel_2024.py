"""Yuksel 2024 rating system.

The rating system from Skill-Based Matchmaking for Competitive Two-Player Games
by Cem Yuksel: http://www.cemyuksel.com/research/matchmaking/

Carries (ratings, R, W, V, D): running weighted mean R, weight W, and variance V
of each competitor's post-match ratings (phi^2 = V / W), plus the recursive
rating-change derivative estimate D. No time dynamics.
"""

import numpy as np
import jax
import jax.numpy as jnp

from riix.utils.constants import Q, Q2_3_OVER_PI2
from riix.core.base import OnlineRatingSystem, _interleave_ratings


class Yuksel2024(OnlineRatingSystem):
    """Yuksel 2024."""

    rating_dim = 2
    _predict_state_slots = (0, 2, 3)  # (ratings, W, V) — phi^2 = V / W

    def __init__(
        self,
        competitors: list,
        initial_rating: float = 1500.0,
        delta_r_max: float = 350.0,
        alpha: float = 2.0,
        scaling_factor: float = 0.9,
        update_method: str = 'online',
        predict_mode: str = 'prequential',
        dtype=jnp.float32,
    ):
        """
        Args:
            initial_rating (float, optional): initial rating for every competitor. Defaults to 1500.0.
            delta_r_max (float, optional): clamp on the per-match rating change. Defaults to 350.0.
            alpha (float, optional): multiplier on the phi (uncertainty) term inside the g function. Defaults to 2.0.
            scaling_factor (float, optional): multiplier on the clamped rating change. Defaults to 0.9.
            update_method / predict_mode / dtype: see `OnlineRatingSystem.__init__`.
        """
        super().__init__(competitors, dtype=dtype, update_method=update_method, predict_mode=predict_mode)
        self.initial_rating = initial_rating
        self.hparams = {
            'q': jnp.asarray(Q, dtype=self.dtype),
            't3q': jnp.asarray(Q2_3_OVER_PI2, dtype=self.dtype),
            'alpha': jnp.asarray(alpha, dtype=self.dtype),
            'scaling_factor': jnp.asarray(scaling_factor, dtype=self.dtype),
            'delta_r_max': jnp.asarray(delta_r_max, dtype=self.dtype),
        }

    def _initial_state(self):
        n = self.num_competitors
        return (
            jnp.full(n, self.initial_rating, dtype=self.dtype),
            jnp.zeros(n, dtype=self.dtype),  # R: weighted mean of post-match ratings
            jnp.full(n, 1e-2, dtype=self.dtype),  # W: running weight
            jnp.zeros(n, dtype=self.dtype),  # V: weighted variance of post-match ratings
            jnp.zeros(n, dtype=self.dtype),  # D: derivative estimate
        )

    @staticmethod
    def _step_core(state, x, hparams):
        ratings, Rs, Ws, Vs, Ds = state
        matchup, outcome = x
        sign_flip = jnp.array([1.0, -1.0], dtype=ratings.dtype)

        r = ratings[matchup]
        R = Rs[matchup]
        W = Ws[matchup]
        V = Vs[matchup]
        D = Ds[matchup]
        pre_W, pre_V = W, V  # emitted pre-update, for period-start predictions and pre-match ratings

        q = hparams['q']
        prob = jax.nn.sigmoid(q * (r[0] - r[1]))

        # V is mathematically non-negative; the max is a float32 backstop against
        # cancellation residuals dipping it below zero (no-op in float64)
        phi = jnp.sqrt(jnp.maximum(V, 0.0) / W)
        g = 1.0 / jnp.sqrt(1.0 + hparams['t3q'] * jnp.square(phi))
        g_alpha = 1.0 / jnp.sqrt(1.0 + hparams['t3q'] * jnp.square(hparams['alpha'] * phi))

        outcome_delta = outcome - prob
        F = jnp.flip(g) * outcome_delta * sign_flip  # (F_1, F_2), each scaled by the opponent's g
        second_deriv = q * prob * (1.0 - prob)
        D = jnp.flip(g) * second_deriv + g_alpha * D

        denom = jnp.maximum(jnp.square(D).sum(), jnp.finfo(ratings.dtype).tiny)
        delta_r = (D * F * sign_flip).sum() / denom
        delta_r = jnp.clip(delta_r, -hparams['delta_r_max'], hparams['delta_r_max'])
        D = jnp.where(delta_r != 0.0, F * sign_flip / delta_r, D)

        new_r = r + hparams['scaling_factor'] * delta_r * sign_flip
        W = g_alpha * W + 1.0
        delta_R = new_r - R
        R = R + delta_R / W
        V = g_alpha * V + delta_R * (new_r - R)

        ratings = ratings.at[matchup].set(new_r)
        Rs = Rs.at[matchup].set(R)
        Ws = Ws.at[matchup].set(W)
        Vs = Vs.at[matchup].set(V)
        Ds = Ds.at[matchup].set(D)
        return (ratings, Rs, Ws, Vs, Ds), prob, r, pre_W, pre_V

    @staticmethod
    def _step(state, x, hparams):
        new_state, prob, _, _, _ = Yuksel2024._step_core(state, x, hparams)
        return new_state, prob

    @staticmethod
    def _step_period_start(state, x, hparams):
        new_state, _, r, W, V = Yuksel2024._step_core(state, x, hparams)
        return new_state, (r, W, V)

    def _gathered_probs(self, gathered):
        r = gathered[0]  # (n, 2) pre-match ratings; W and V don't enter the prediction
        return jax.nn.sigmoid(self.hparams['q'] * (r[:, 0] - r[:, 1]))

    def _gathered_ratings(self, gathered):
        r, W, V = gathered
        phi = jnp.sqrt(jnp.maximum(V, 0.0) / W)  # same uncertainty the update's g functions use
        return _interleave_ratings((r, phi))  # columns are (rating, phi), like the Glicko family

    @property
    def ratings(self):
        return np.asarray(self.state[0])

    @property
    def R(self):
        return np.asarray(self.state[1])

    @property
    def W(self):
        return np.asarray(self.state[2])

    @property
    def V(self):
        return np.asarray(self.state[3])

    @property
    def D(self):
        return np.asarray(self.state[4])

    def leaderboard_ratings(self):
        return self.ratings
