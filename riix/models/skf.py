"""Vector covariance simplified Kalman filter (VSKF) rating system.

`model` selects the online update moments like v1: 'bt' (Bradley-Terry, base-10
sigmoid) or 'tm' (Thurstone-Mosteller, normal cdf with TrueSkill-style v/w).

Unlike the Glicko family's additive inflation, VSKF's per-period time dynamics
are affine: mu <- beta^dt * mu and v <- beta^(2*dt) * v + dt * epsilon, so the
epsilon injections decay under later dampening and a summed dt alone cannot
reproduce an idle span. Affine maps compose into affine maps, though: with G_k
the noise a competitor active since the start would hold after period k
(a scan of the per-period steps from 0), an idle span from period a to period p
is exactly v <- beta^(2*(T_p - T_a)) * (v - G_a) + G_p. G is computed inside
jit from the traced beta/epsilon (sweeps still never recompile); the numpy
pre-pass only emits integer indices into it.
"""

import math
import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.stats import norm

from riix.core.base import OnlineRatingSystem, compute_inflation_dt

LOG10 = math.log(10.0)
LOG10_SQUARED = LOG10**2.0


def compute_dynamics_indices(
    matchups: np.ndarray,
    time_steps: np.ndarray,
    num_competitors: int,
    initial_prev_time_step: int = 0,
    has_played: np.ndarray = None,
):
    """
    Indices into the accumulated-noise series G for each competitor-occurrence,
    aligned with the (dt_pre, dt_post) split of `compute_inflation_dt`.

    G-index k means "after the dynamics of the k-th rating period containing
    matches" (0 = before any — i.e. the state as of `initial_prev_time_step`).
    Per occurrence, [..., 0] is G at the competitor's previous appearance and
    [..., 1] is G at the period before the match's own; on a first or
    same-period repeat appearance both are equal so the noise term vanishes,
    exactly when dt_pre is 0.

    For continuation (`has_played`), a previous fit's final dynamics left every
    such competitor evolved exactly through `initial_prev_time_step` — G-index 0
    — so their first appearance here resumes from there, and the ones absent
    from this dataset owe the whole span at fit end.

    Returns:
        idx: (n, 2, 2) int64 G-indices aligned with matchups.
        final_idx: (num_competitors,) G-index of each competitor's last
            appearance (num_periods for competitors who never appear, so the
            fit-end noise term vanishes together with final_dt = 0).
        period_deltas: (num_periods,) float64 gaps between consecutive rating
            periods (the first relative to `initial_prev_time_step`), the xs
            of the G recursion.
    """
    n = matchups.shape[0]
    if n == 0:
        return np.zeros((0, 2, 2), dtype=np.int64), np.zeros(num_competitors, dtype=np.int64), np.zeros(0)

    unique_ts = np.unique(np.asarray(time_steps, dtype=np.int64))
    num_periods = len(unique_ts)
    period_deltas = np.diff(unique_ts, prepend=np.int64(initial_prev_time_step)).astype(np.float64)

    flat_comp = matchups.reshape(-1)
    flat_ts = np.repeat(np.asarray(time_steps, dtype=np.int64), 2)
    # stable sort by competitor keeps each competitor's appearances in match order
    order = np.argsort(flat_comp, kind='stable')
    comp_sorted = flat_comp[order]
    period_idx = np.searchsorted(unique_ts, flat_ts[order])

    prev_idx = np.empty_like(period_idx)
    prev_idx[1:] = period_idx[:-1]
    prev_idx[0] = 0
    is_first = np.empty(comp_sorted.shape, dtype=bool)
    is_first[0] = True
    is_first[1:] = comp_sorted[1:] != comp_sorted[:-1]
    # previous appearance in period a -> G-index a+1; same-period repeat (a == b)
    # collapses onto the match's own index so the noise term is 0
    g_prev = np.where(is_first, period_idx, np.minimum(prev_idx + 1, period_idx))
    if has_played is not None:
        # continuation: competitors from earlier fits resume from G-index 0
        g_prev[is_first & has_played[comp_sorted]] = 0

    idx = np.empty((2 * n, 2), dtype=np.int64)
    idx[order, 0] = g_prev
    idx[order, 1] = period_idx

    is_last = np.empty(comp_sorted.shape, dtype=bool)
    is_last[-1] = True
    is_last[:-1] = comp_sorted[1:] != comp_sorted[:-1]
    final_idx = np.full(num_competitors, num_periods, dtype=np.int64)
    if has_played is not None:
        final_idx[has_played] = 0
    final_idx[comp_sorted[is_last]] = period_idx[is_last] + 1
    return idx.reshape(n, 2, 2), final_idx, period_deltas


@jax.jit
def _accumulated_noise(period_deltas, beta, epsilon):
    """G_k for k in 0..num_periods: process noise accumulated through period k from 0."""

    def body(g, d):
        g = jnp.square(beta**d) * g + epsilon * d
        return g, g

    zero = jnp.zeros((), dtype=period_deltas.dtype)
    _, gs = jax.lax.scan(body, zero, period_deltas)
    return jnp.concatenate([zero[None], gs])


class VSKF(OnlineRatingSystem):
    """Vector covariance simplified Kalman filter."""

    rating_dim = 2
    leaderboard_header = 'mu - (3 * sd)'
    uses_dt = True
    initial_prev_time_step = 0
    _predict_state_slots = (0, 1)  # (mus, variances)

    def __init__(
        self,
        competitors: list,
        mu_0: float = 0.0,
        v_0: float = 1.0,
        beta: float = 1.0,
        s: float = 1.0,
        epsilon: float = 1e-2,
        model='bt',
        update_method: str = 'online',
        predict_mode: str = 'prequential',
        dtype=jnp.float32,
    ):
        """
        Args:
            mu_0 (float, optional): initial skill mean for every competitor. Defaults to 0.0.
            v_0 (float, optional): initial skill variance for every competitor. Defaults to 1.0.
            beta (float, optional): per-period dampening of the skill mean and variance toward the prior (1.0 disables dampening). Defaults to 1.0.
            s (float, optional): performance noise scale in the observation model. Defaults to 1.0.
            epsilon (float, optional): process noise added to the variance per rating period. Defaults to 1e-2.
            model (str, optional): observation model — 'bt' (Bradley-Terry / logistic) or 'tm' (Thurstone-Mosteller / probit). Defaults to 'bt'.
            update_method / predict_mode / dtype: see `OnlineRatingSystem.__init__`.
        """
        super().__init__(competitors, dtype=dtype, update_method=update_method, predict_mode=predict_mode)
        self.mu_0 = mu_0
        self.v_0 = v_0
        self.model = model
        self.hparams = {
            'beta': jnp.asarray(beta, dtype=self.dtype),
            's': jnp.asarray(s, dtype=self.dtype),
            's2': jnp.asarray(s**2.0, dtype=self.dtype),
            'epsilon': jnp.asarray(epsilon, dtype=self.dtype),
        }
        if model == 'bt':
            self._step = VSKF._step_bt
            self._step_period_start = VSKF._step_period_start_bt
            self._batched_step = VSKF._batched_step_bt
        elif model == 'tm':
            self._step = VSKF._step_tm
            self._step_period_start = VSKF._step_period_start_tm
            self._batched_step = VSKF._batched_step_tm
        else:
            raise ValueError(f'Invalid model {model}')

    def _initial_state(self):
        return (
            jnp.full(self.num_competitors, self.mu_0, dtype=self.dtype),
            jnp.full(self.num_competitors, self.v_0, dtype=self.dtype),
        )

    def _prepare_online_xs(self, dataset, matchups, outcomes):
        prev_time_step = self._continuation_prev_time_step()
        schedule = compute_inflation_dt(
            dataset.matchups,
            dataset.time_steps,
            prev_time_step,
            self.num_competitors,
            has_played=self._has_played,
        )
        idx, final_idx, period_deltas = compute_dynamics_indices(
            dataset.matchups,
            dataset.time_steps,
            self.num_competitors,
            initial_prev_time_step=prev_time_step,
            has_played=self._has_played,
        )
        noise = _accumulated_noise(
            jnp.asarray(period_deltas, dtype=self.dtype), self.hparams['beta'], self.hparams['epsilon']
        )
        step_xs = (matchups, outcomes, jnp.asarray(schedule.dts, dtype=self.dtype), noise[idx])
        final_args = (jnp.asarray(schedule.final_dt, dtype=self.dtype), noise[final_idx], noise[-1])
        return step_xs, final_args

    # ------------------------------------------------------------ online steps

    @staticmethod
    def _gather_and_evolve(state, x, hparams):
        """Gather the two competitors and apply the lazy affine time dynamics split."""
        mus, vs = state
        matchup, outcome, dt, g = x
        beta = hparams['beta']
        mu = mus[matchup]
        v = vs[matchup]

        # idle dynamics through the previous rating period (visible to predictions);
        # the noise increment is formed before touching v so its rounding stays at
        # the increment's magnitude, not G's
        decay_pre = beta ** dt[:, 0]
        decay2_pre = jnp.square(decay_pre)
        mu = mu * decay_pre
        v = jnp.where(dt[:, 0] > 0.0, decay2_pre * v + (g[:, 1] - decay2_pre * g[:, 0]), v)
        mu_pred = mu  # what a (period-start) prediction sees, before this period's dynamics
        v_pred = v

        # this period's dynamics: a single per-period step, applied post-prediction
        decay_post = beta ** dt[:, 1]
        mu = mu * decay_post
        v = jnp.square(decay_post) * v + dt[:, 1] * hparams['epsilon']

        z = (mu[0] - mu[1]) / hparams['s']
        z_pred = (mu_pred[0] - mu_pred[1]) / hparams['s']
        return mu, v, z, z_pred, mu_pred, v_pred, outcome, matchup

    @staticmethod
    def _apply(state, matchup, mu, v, grad, hess, hparams):
        mus, vs = state
        omega = v[0] + v[1]
        denom = hparams['s2'] + hess * omega
        mu_update = (hparams['s'] * grad) / denom
        v_update = hess / denom
        mu_new = mu + v * mu_update * jnp.array([1.0, -1.0], dtype=mu.dtype)
        v_new = v * (1.0 - v * v_update)
        return (mus.at[matchup].set(mu_new), vs.at[matchup].set(v_new))

    @staticmethod
    def _step_core_bt(state, x, hparams):
        mu, v, z, z_pred, mu_pred, v_pred, outcome, matchup = VSKF._gather_and_evolve(state, x, hparams)
        prob = jax.nn.sigmoid(LOG10 * z_pred)  # matches v1 predict() for model='bt'
        p = jax.nn.sigmoid(LOG10 * z)
        grad = LOG10 * (outcome - p)
        hess = LOG10_SQUARED * p * (1.0 - p)
        return VSKF._apply(state, matchup, mu, v, grad, hess, hparams), prob, mu_pred, v_pred

    @staticmethod
    def _step_core_tm(state, x, hparams):
        mu, v, z, z_pred, mu_pred, v_pred, outcome, matchup = VSKF._gather_and_evolve(state, x, hparams)
        # float32 guard floors, as in v2 TrueSkill (no-ops in float64)
        tiny = float(jnp.finfo(mu.dtype).tiny)
        win_guard = max(2.222758749e-162, tiny)

        sign_multiplier = (2.0 * outcome) - 1.0  # 0 for draws, whose moments come from the draw branch below
        t = z * sign_multiplier
        cdfs = norm.cdf(jnp.stack([z_pred, t]))
        prob = cdfs[0]  # matches v1 predict() for model='tm'
        cdf_win = cdfs[1]
        v_raw = jnp.where(cdf_win > win_guard, norm.pdf(t) / jnp.maximum(cdf_win, tiny), -t)
        w_win = v_raw * (v_raw + t)

        # v1 calls v_and_w_draw_scalar(z, eps=0.0): a zero draw margin makes its shared
        # denominator exactly 0, so it always takes the guard branch: v = -z, w = 1
        is_draw = outcome == 0.5
        grad = jnp.where(is_draw, -z, v_raw * sign_multiplier)
        # w is a variance-reduction fraction, theoretically in (0, 1]; float32 backstop
        hess = jnp.clip(jnp.where(is_draw, 1.0, w_win), 0.0, 1.0)
        return VSKF._apply(state, matchup, mu, v, grad, hess, hparams), prob, mu_pred, v_pred

    @staticmethod
    def _step_bt(state, x, hparams):
        new_state, prob, _, _ = VSKF._step_core_bt(state, x, hparams)
        return new_state, prob

    @staticmethod
    def _step_tm(state, x, hparams):
        new_state, prob, _, _ = VSKF._step_core_tm(state, x, hparams)
        return new_state, prob

    @staticmethod
    def _step_period_start_bt(state, x, hparams):
        new_state, _, mu_pred, v_pred = VSKF._step_core_bt(state, x, hparams)
        return new_state, (mu_pred, v_pred)

    @staticmethod
    def _step_period_start_tm(state, x, hparams):
        new_state, _, mu_pred, v_pred = VSKF._step_core_tm(state, x, hparams)
        return new_state, (mu_pred, v_pred)

    def _gathered_probs(self, gathered):
        mu = gathered[0]  # (n, 2) pre-match mus; the variances don't enter the prediction
        z = (mu[:, 0] - mu[:, 1]) / self.hparams['s']
        return jax.nn.sigmoid(LOG10 * z) if self.model == 'bt' else norm.cdf(z)

    # ------------------------------------------------------------ batched mode

    def _initial_state_batched(self):
        # the batched carry additionally tracks who has ever played, since idle
        # dynamics are applied eagerly per period rather than lazily per match
        return (*self._initial_state(), jnp.zeros(self.num_competitors, dtype=bool))

    @staticmethod
    def _batched_core(state, x, hparams):
        """One padded rating period, mirroring v1 VSKF.batched_update.

        The pooled update is model-independent in v1 (it always uses the
        Bradley-Terry gradients, even for model='tm'); only the returned
        predictions differ per model, so this emits the period-start rating
        diffs for the wrappers to turn into probabilities.
        """
        mus, vs, has_played = state
        matchups, outcomes, mask, time_delta = x
        flat_idx = matchups.reshape(-1)
        mask2 = mask[:, None]

        # returned predictions come from period-start, pre-dynamics state
        # (v1 calls predict() before batched_update)
        mu_pre = mus[matchups]  # (p, 2)
        pred_zs = (mu_pre[:, 0] - mu_pre[:, 1]) / hparams['s']

        # brand-new actives get the current period's single unit of dynamics, not the global gap
        active = jnp.zeros_like(mus).at[flat_idx].max(jnp.repeat(mask, 2)) > 0.0
        dt_eff = jnp.where(has_played, time_delta, jnp.minimum(time_delta, 1.0))
        has_played = has_played | active
        decay = hparams['beta'] ** dt_eff
        mus = jnp.where(has_played, decay * mus, mus)
        vs = jnp.where(has_played, jnp.square(decay) * vs + dt_eff * hparams['epsilon'], vs)

        mu = mus[matchups]
        v = vs[matchups]
        omega = v[:, 0] + v[:, 1]
        p = jax.nn.sigmoid(LOG10 * (mu[:, 0] - mu[:, 1]) / hparams['s'])
        grad = LOG10 * (outcomes - p)
        hess = LOG10_SQUARED * p * (1.0 - p)
        denom = hparams['s2'] + hess * omega
        mu_updates = ((hparams['s'] * grad) / denom)[:, None] * jnp.array([1.0, -1.0], dtype=mus.dtype)
        multipliers = 1.0 - v * (hess / denom)[:, None]

        mu_pooled = jnp.zeros_like(mus).at[flat_idx].add((mu_updates * mask2).reshape(-1))
        # v1 pools the v multipliers with a zero-floored scatter-max; padding rows
        # contribute exactly the floor
        mult_pooled = jnp.zeros_like(vs).at[flat_idx].max((multipliers * mask2).reshape(-1))

        mus = mus + jnp.where(active, vs * mu_pooled, 0.0)
        vs = jnp.where(active, vs * mult_pooled, vs)
        return (mus, vs, has_played), pred_zs

    @staticmethod
    def _batched_step_bt(state, x, hparams):
        new_state, pred_zs = VSKF._batched_core(state, x, hparams)
        return new_state, jax.nn.sigmoid(LOG10 * pred_zs)

    @staticmethod
    def _batched_step_tm(state, x, hparams):
        new_state, pred_zs = VSKF._batched_core(state, x, hparams)
        return new_state, norm.cdf(pred_zs)

    def _apply_final_inflation(self, state, final_args):
        final_dt, noise_last, noise_end = final_args
        mus, vs = state
        decay = self.hparams['beta'] ** final_dt
        decay2 = jnp.square(decay)
        mus = mus * decay
        vs = jnp.where(final_dt > 0.0, decay2 * vs + (noise_end - decay2 * noise_last), vs)
        return (mus, vs)

    @property
    def mus(self):
        return np.asarray(self.state[0])

    @property
    def vs(self):
        return np.asarray(self.state[1])

    def leaderboard_ratings(self):
        return self.mus - (3.0 * np.sqrt(self.vs))
