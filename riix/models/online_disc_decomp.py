"""Online version of the Disc Decomposition rating system
https://proceedings.mlr.press/v206/bertrand23a.html
equations 8 and 9"""

import numpy as np
import jax
import jax.numpy as jnp

from riix.core.base import OnlineRatingSystem


class OnlineDiscDecomp(OnlineRatingSystem):
    """Online Disc Decomposition.

    Carries (us, vs); P(1 beats 2) = sigmoid(v1 * v2 * (u1 - u2)). No time dynamics.
    """

    rating_dim = 2
    _predict_state_slots = (0, 1)  # (us, vs)

    def __init__(
        self,
        competitors: list,
        initial_u: float = 1.0,
        initial_v: float = 1.0,
        eta: float = 0.02,
        update_method: str = 'online',
        predict_mode: str = 'prequential',
        dtype=jnp.float32,
    ):
        """
        Args:
            initial_u (float, optional): initial u (skill) value for every competitor. Defaults to 1.0.
            initial_v (float, optional): initial v (consistency) value for every competitor. Defaults to 1.0.
            eta (float, optional): gradient step size. Defaults to 0.02.
            update_method / predict_mode / dtype: see `OnlineRatingSystem.__init__`.
        """
        super().__init__(competitors, dtype=dtype, update_method=update_method, predict_mode=predict_mode)
        self.initial_u = initial_u
        self.initial_v = initial_v
        self.hparams = {'eta': jnp.asarray(eta, dtype=self.dtype)}

    def _initial_state(self):
        return (
            jnp.full(self.num_competitors, self.initial_u, dtype=self.dtype),
            jnp.full(self.num_competitors, self.initial_v, dtype=self.dtype),
        )

    @staticmethod
    def _step_core(state, x, hparams):
        us, vs = state
        matchup, outcome = x
        u = us[matchup]
        v = vs[matchup]
        u_diff = u[0] - u[1]
        v_prod = v[0] * v[1]
        prob = jax.nn.sigmoid(v_prod * u_diff)
        scaled_delta = hparams['eta'] * (outcome - prob)

        # paper equation 8: u_i += eta * delta * v_i * v_j, u_j gets the negated update
        u_update = scaled_delta * v_prod
        # paper equation 9: v_i += eta * delta * v_j * (u_i - u_j), same sign for both sides
        v_update = scaled_delta * u_diff

        us = us.at[matchup].add(jnp.stack([u_update, -u_update]))
        vs = vs.at[matchup].add(v_update * jnp.flip(v))
        return (us, vs), prob, u, v

    @staticmethod
    def _step(state, x, hparams):
        new_state, prob, _, _ = OnlineDiscDecomp._step_core(state, x, hparams)
        return new_state, prob

    @staticmethod
    def _step_period_start(state, x, hparams):
        new_state, _, u, v = OnlineDiscDecomp._step_core(state, x, hparams)
        return new_state, (u, v)

    def _gathered_probs(self, gathered):
        u, v = gathered  # (n, 2) pre-match us and vs
        return jax.nn.sigmoid(v[:, 0] * v[:, 1] * (u[:, 0] - u[:, 1]))

    @staticmethod
    def _batched_step(state, x, hparams):
        """One padded rating period, mirroring v1 OnlineDiscDecomp.batched_update."""
        us, vs = state
        matchups, outcomes, mask, _time_delta = x
        flat_idx = matchups.reshape(-1)

        u = us[matchups]  # (p, 2)
        v = vs[matchups]
        u_diffs = u[:, 0] - u[:, 1]
        v_prods = v[:, 0] * v[:, 1]
        probs = jax.nn.sigmoid(v_prods * u_diffs)
        scaled_deltas = hparams['eta'] * (outcomes - probs) * mask

        u_updates = (scaled_deltas * v_prods)[:, None] * jnp.array([1.0, -1.0], dtype=us.dtype)
        v_updates = (scaled_deltas * u_diffs)[:, None] * v[:, ::-1]

        us = us.at[flat_idx].add(u_updates.reshape(-1))
        vs = vs.at[flat_idx].add(v_updates.reshape(-1))
        return (us, vs), probs

    @property
    def us(self):
        return np.asarray(self.state[0])

    @property
    def vs(self):
        return np.asarray(self.state[1])

    def leaderboard_ratings(self):
        return self.us
