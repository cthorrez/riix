"""Temporalized Massey Method
https://www.degruyter.com/document/doi/10.1515/jqas-2016-0093
https://arxiv.org/abs/1702.00585"""

import numpy as np
import jax.numpy as jnp

from riix.core.base import OnlineRatingSystem


class TemporalMassey(OnlineRatingSystem):
    """Temporalized Massey method.

    Passing `alpha=None` or `beta=None` selects v1's varying-coefficient mode, where
    each competitor's coefficients derive from their running matchup count (carried
    in the scan state instead of being hyperparameters).
    """

    rating_dim = 1

    def __init__(
        self,
        competitors: list,
        initial_rating: float = 1.0,
        alpha: float = 0.95,
        beta: float = 0.05,
        update_method: str = 'online',
        predict_mode: str = 'prequential',
        dtype=jnp.float32,
    ):
        """
        Args:
            initial_rating (float, optional): initial rating for every competitor. Defaults to 1.0.
            alpha (float or None, optional): retention weight on a competitor's own previous rating; None selects the varying-coefficient mode where per-competitor weights are derived from matchup counts. Defaults to 0.95.
            beta (float or None, optional): weight on the match adjustment; None likewise selects the varying-coefficient mode. Defaults to 0.05.
            update_method / predict_mode / dtype: see `OnlineRatingSystem.__init__`.
        """
        super().__init__(competitors, dtype=dtype, update_method=update_method, predict_mode=predict_mode)
        self.initial_rating = initial_rating
        self.varying_coefs = (alpha is None) or (beta is None)
        if self.varying_coefs:
            self.hparams = {}
            self._step = TemporalMassey._step_varying
            self._step_period_start = TemporalMassey._step_period_start_varying
        else:
            self.hparams = {
                'alpha': jnp.asarray(alpha, dtype=self.dtype),
                'beta': jnp.asarray(beta, dtype=self.dtype),
            }

    def _initial_state(self):
        ratings = jnp.full(self.num_competitors, self.initial_rating, dtype=self.dtype)
        if self.varying_coefs:
            return (ratings, jnp.zeros(self.num_competitors, dtype=self.dtype))
        return (ratings,)

    @staticmethod
    def _apply_update(ratings, matchup, outcome, r, a, b):
        s_1 = (outcome - 0.5) * 2.0
        r_1_new = (a[0] * r[0]) + (b[0] * r[1]) + (b[0] * s_1)
        r_2_new = (a[1] * r[1]) + (b[1] * r[0]) - (b[1] * s_1)
        return ratings.at[matchup].set(jnp.stack([r_1_new, r_2_new]))

    @staticmethod
    def _step_core(state, x, hparams):
        (ratings,) = state
        matchup, outcome = x
        r = ratings[matchup]
        prob = r[0] / (r[0] + r[1])
        a = jnp.stack([hparams['alpha'], hparams['alpha']])
        b = jnp.stack([hparams['beta'], hparams['beta']])
        ratings = TemporalMassey._apply_update(ratings, matchup, outcome, r, a, b)
        return (ratings,), prob, r

    @staticmethod
    def _step(state, x, hparams):
        new_state, prob, _ = TemporalMassey._step_core(state, x, hparams)
        return new_state, prob

    @staticmethod
    def _step_period_start(state, x, hparams):
        new_state, _, r = TemporalMassey._step_core(state, x, hparams)
        return new_state, r

    @staticmethod
    def _step_core_varying(state, x, hparams):
        ratings, counts = state
        matchup, outcome = x
        r = ratings[matchup]
        prob = r[0] / (r[0] + r[1])
        cnt = counts[matchup]
        safe = jnp.maximum(cnt, 1.0)
        a = jnp.where(cnt > 0.0, (cnt - 1.0) / safe, 0.0)
        b = jnp.where(cnt > 0.0, 1.0 / safe, 0.0)
        ratings = TemporalMassey._apply_update(ratings, matchup, outcome, r, a, b)
        counts = counts.at[matchup].add(1.0)
        return (ratings, counts), prob, r

    @staticmethod
    def _step_varying(state, x, hparams):
        new_state, prob, _ = TemporalMassey._step_core_varying(state, x, hparams)
        return new_state, prob

    @staticmethod
    def _step_period_start_varying(state, x, hparams):
        new_state, _, r = TemporalMassey._step_core_varying(state, x, hparams)
        return new_state, r

    def _gathered_probs(self, gathered):
        (r,) = gathered  # (n, 2) pre-match ratings
        return r[:, 0] / (r[:, 0] + r[:, 1])

    @property
    def ratings(self):
        return np.asarray(self.state[0])

    @property
    def matchup_counts(self):
        if not self.varying_coefs:
            raise AttributeError('matchup_counts only exists in varying-coefficient mode (alpha=None or beta=None)')
        return np.asarray(self.state[1])

    def leaderboard_ratings(self):
        return self.ratings
