"""Glicko rating system.

Rating deviation is carried as a variance (RD^2) so the per-match update never
needs a sqrt round-trip; RDs are only materialized on demand. Time-based RD
inflation uses per-appearance elapsed periods precomputed in numpy (see
`riix.core.base.compute_inflation_dt`), so the scan carry is just two float
arrays.
"""

import numpy as np
import jax
import jax.numpy as jnp

from riix.utils.constants import Q, Q2, Q2_3_OVER_PI2
from riix.core.base import OnlineRatingSystem, _interleave_ratings


class Glicko(OnlineRatingSystem):
    """Mark Glickman's Glicko rating system."""

    rating_dim = 2
    leaderboard_header = 'rating - (3*dev)'
    uses_dt = True
    initial_prev_time_step = -1
    _predict_state_slots = (0, 1)  # (ratings, RD^2)

    def __init__(
        self,
        competitors: list,
        initial_rating: float = 1500.0,
        initial_rating_dev: float = 350.0,
        c: float = 63.2,
        update_method: str = 'online',
        predict_mode: str = 'prequential',
        dtype=jnp.float32,
    ):
        """
        Args:
            initial_rating (float, optional): The initial Glicko rating for new competitors. Defaults to 1500.0.
            initial_rating_dev (float, optional): The initial rating deviation for new competitors. Defaults to 350.0.
            c (float, optional): Constant used to adjust the rate of change of the rating deviation. Defaults to 63.2.
            update_method / predict_mode / dtype: see `OnlineRatingSystem.__init__`.
        """
        super().__init__(competitors, dtype=dtype, update_method=update_method, predict_mode=predict_mode)
        self.initial_rating = initial_rating
        self.initial_rating_dev = initial_rating_dev
        self.hparams = {
            'c2': jnp.asarray(c**2.0, dtype=self.dtype),
            'max_rd2': jnp.asarray(initial_rating_dev**2.0, dtype=self.dtype),
            'q': jnp.asarray(Q, dtype=self.dtype),
            'q2': jnp.asarray(Q2, dtype=self.dtype),
            'tq2_pi2': jnp.asarray(Q2_3_OVER_PI2, dtype=self.dtype),
        }

    def _initial_state(self):
        return (
            jnp.full(self.num_competitors, self.initial_rating, dtype=self.dtype),
            jnp.full(self.num_competitors, self.initial_rating_dev**2.0, dtype=self.dtype),  # RD^2
        )

    @staticmethod
    def _step_core(state, x, hparams):
        ratings, rd2s = state
        matchup, outcome, dt = x
        sign_flip = jnp.array([1.0, -1.0], dtype=ratings.dtype)

        r = ratings[matchup]
        w = rd2s[matchup]

        # lazy RD inflation, capped at the initial RD; dt[:, 0] is the part
        # accumulated before this rating period (visible to the prediction),
        # dt[:, 1] is this period's inflation (applied only for the update)
        w = jnp.minimum(w + hparams['c2'] * dt[:, 0], hparams['max_rd2'])
        w_pred = w  # what a period-start prediction sees (see _gathered_probs)

        # matches v1 predict(): sigmoid(q * g(sqrt(rd1^2 + rd2^2)) * rating_diff)
        g_comb = 1.0 / jnp.sqrt(1.0 + (w[0] + w[1]) * hparams['tq2_pi2'])
        prob = jax.nn.sigmoid(hparams['q'] * g_comb * (r[0] - r[1]))

        w = jnp.minimum(w + hparams['c2'] * dt[:, 1], hparams['max_rd2'])

        # each competitor's update uses the opponent's g
        g_opp = jnp.flip(1.0 / jnp.sqrt(1.0 + w * hparams['tq2_pi2']))
        diffs = (r * sign_flip).sum() * sign_flip  # (r1 - r2, r2 - r1)
        expected = jax.nn.sigmoid(hparams['q'] * g_opp * diffs)
        d2_inv = hparams['q2'] * jnp.square(g_opp) * expected * (1.0 - expected)

        outcomes_both = jnp.stack([outcome, 1.0 - outcome])
        denom = (1.0 / w) + d2_inv
        r_new = r + hparams['q'] * g_opp * (outcomes_both - expected) / denom

        ratings = ratings.at[matchup].set(r_new)
        rd2s = rd2s.at[matchup].set(1.0 / denom)
        return (ratings, rd2s), prob, r, w_pred

    @staticmethod
    def _step(state, x, hparams):
        new_state, prob, _, _ = Glicko._step_core(state, x, hparams)
        return new_state, prob

    @staticmethod
    def _step_period_start(state, x, hparams):
        new_state, _, r, w_pred = Glicko._step_core(state, x, hparams)
        return new_state, (r, w_pred)

    def _gathered_probs(self, gathered):
        r, w = gathered  # (n, 2) pre-match ratings and (inflated, capped) RD^2
        g_comb = 1.0 / jnp.sqrt(1.0 + (w[:, 0] + w[:, 1]) * self.hparams['tq2_pi2'])
        return jax.nn.sigmoid(self.hparams['q'] * g_comb * (r[:, 0] - r[:, 1]))

    def _gathered_ratings(self, gathered):
        r, w = gathered
        return _interleave_ratings((r, jnp.sqrt(w)))  # v1 columns are (rating, rating_dev)

    def _initial_state_batched(self):
        # the batched carry additionally tracks who has ever played, since idle
        # inflation is applied eagerly per period rather than lazily per match
        return (*self._initial_state(), jnp.zeros(self.num_competitors, dtype=bool))

    @staticmethod
    def _batched_step(state, x, hparams):
        """One padded rating period, mirroring v1 Glicko.batched_update."""
        ratings, rd2s, has_played = state
        matchups, outcomes, mask, time_delta = x
        flat_idx = matchups.reshape(-1)
        zeros = jnp.zeros(ratings.shape[0], dtype=ratings.dtype)

        r = ratings[matchups]  # (p, 2)
        w_pre = rd2s[matchups]

        # returned predictions, from period-start state (v1 calls predict() first)
        g_comb = 1.0 / jnp.sqrt(1.0 + (w_pre[:, 0] + w_pre[:, 1]) * hparams['tq2_pi2'])
        probs = jax.nn.sigmoid(hparams['q'] * g_comb * (r[:, 0] - r[:, 1]))

        # capped inflation for everyone who has ever played, active included
        # (unlike Glicko2's batched update there is no separate phi_star path);
        # brand-new actives get the current period's single unit, not the global gap
        active = zeros.at[flat_idx].max(jnp.repeat(mask, 2)) > 0.0
        dt_eff = jnp.where(has_played, time_delta, jnp.minimum(time_delta, 1.0))
        has_played = has_played | active
        rd2s = jnp.where(has_played, jnp.minimum(rd2s + hparams['c2'] * dt_eff, hparams['max_rd2']), rd2s)

        w = rd2s[matchups]
        g_opp = (1.0 / jnp.sqrt(1.0 + w * hparams['tq2_pi2']))[:, ::-1]
        diffs = r - r[:, ::-1]  # (p, 2): (r1 - r2, r2 - r1)
        expected = jax.nn.sigmoid(hparams['q'] * g_opp * diffs)
        outcomes_both = jnp.stack([outcomes, 1.0 - outcomes], axis=1)
        mask2 = mask[:, None]

        d2_inv_sum = zeros.at[flat_idx].add((jnp.square(g_opp) * expected * (1.0 - expected) * mask2).reshape(-1))
        grads = zeros.at[flat_idx].add((g_opp * (outcomes_both - expected) * mask2).reshape(-1))

        denom = (1.0 / rd2s) + hparams['q2'] * d2_inv_sum
        ratings = ratings + jnp.where(active, hparams['q'] * grads / denom, 0.0)
        rd2s = jnp.where(active, 1.0 / denom, rd2s)
        return (ratings, rd2s, has_played), probs

    def _apply_final_inflation(self, state, final_dt):
        ratings, rd2s = state
        rd2s = jnp.minimum(rd2s + self.hparams['c2'] * final_dt, self.hparams['max_rd2'])
        return (ratings, rd2s)

    @property
    def ratings(self):
        state = self.state if self.state is not None else self._initial_state()
        return np.asarray(state[0])

    @ratings.setter
    def ratings(self, values):
        self._set_state_slot(0, values)

    @property
    def rating_devs(self):
        state = self.state if self.state is not None else self._initial_state()
        return np.sqrt(np.asarray(state[1]))

    @rating_devs.setter
    def rating_devs(self, values):
        self._set_state_slot(1, np.square(np.asarray(values)))  # carried as RD^2

    def leaderboard_ratings(self):
        return self.ratings - 3.0 * self.rating_devs
