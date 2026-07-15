"""Weng-Lin Bayesian online rating system rating system.

`model` selects the update math like v1: 'bt' (Bradley-Terry, sigmoid) or 'tm'
(Thurstone-Mosteller, normal cdf with the TrueSkill-style v/w moments). Both
share the additive tau^2 variance inflation handled by the dt pre-pass
(`prev_time_step` starts at 0, like TrueSkill).
"""

import math
import numpy as np
from scipy.stats import norm as scipy_norm
import jax
import jax.numpy as jnp
from jax.scipy.stats import norm

from riix.core.base import OnlineRatingSystem


class WengLin(OnlineRatingSystem):
    """Weng-Lin."""

    rating_dim = 2
    leaderboard_header = 'mu - (3 * sd)'
    uses_dt = True
    initial_prev_time_step = 0  # v1 WengLin starts prev_time_step at 0, like TrueSkill
    _predict_state_slots = (0, 1)  # (mus, sigma^2)

    def __init__(
        self,
        competitors: list,
        initial_mu: float = 25.0,
        initial_sigma: float = 8.333,
        beta: float = 4.166,
        kappa: float = 0.0001,
        tau: float = 0.0833,
        draw_probability=0.0,
        update_method: str = 'online',
        predict_mode: str = 'prequential',
        model='tm',
        dtype=jnp.float32,
    ):
        """
        Args:
            initial_mu (float, optional): initial skill mean for every competitor. Defaults to 25.0.
            initial_sigma (float, optional): initial skill standard deviation. Defaults to 8.333.
            beta (float, optional): performance noise standard deviation. Defaults to 4.166.
            kappa (float, optional): floor on the variance-update factor, keeping variances positive. Defaults to 1e-4.
            tau (float, optional): additive skill dynamics noise per rating period. Defaults to 0.0833.
            draw_probability (float, optional): assumed probability of a draw, setting the draw margin. Defaults to 0.0.
            model (str, optional): update moments — 'bt' (Bradley-Terry / logistic) or 'tm' (Thurstone-Mosteller / probit). Defaults to 'tm'.
            update_method / predict_mode / dtype: see `OnlineRatingSystem.__init__`.
        """
        super().__init__(competitors, dtype=dtype, update_method=update_method, predict_mode=predict_mode)
        self.initial_mu = initial_mu
        self.initial_sigma = initial_sigma
        self.model = model
        epsilon = scipy_norm.ppf((draw_probability + 1.0) / 2.0) * math.sqrt(2.0) * beta
        self.hparams = {
            'two_beta2': jnp.asarray(2.0 * beta**2.0, dtype=self.dtype),
            'tau2': jnp.asarray(tau**2.0, dtype=self.dtype),
            'kappa': jnp.asarray(kappa, dtype=self.dtype),
            'epsilon': jnp.asarray(epsilon, dtype=self.dtype),
        }
        if model == 'bt':
            self._step = WengLin._step_bt
            self._step_period_start = WengLin._step_period_start_bt
            self._batched_step = WengLin._batched_step_bt
        elif model == 'tm':
            self._step = WengLin._step_tm
            self._step_period_start = WengLin._step_period_start_tm
            self._batched_step = WengLin._batched_step_tm
        else:
            raise ValueError(f'Invalid model {model}')

    def _initial_state(self):
        return (
            jnp.full(self.num_competitors, self.initial_mu, dtype=self.dtype),
            jnp.full(self.num_competitors, self.initial_sigma**2.0, dtype=self.dtype),
        )

    # ------------------------------------------------------------ online steps

    @staticmethod
    def _gather_and_inflate(state, x, hparams):
        """Gather the two competitors and apply the lazy variance inflation split."""
        mus, sigma2s = state
        matchup, outcome, dt = x
        mu = mus[matchup]
        s2 = sigma2s[matchup] + dt[:, 0] * hparams['tau2']
        s2_pred = s2  # what a (period-start) prediction sees, before this period's inflation
        pred_norm_diff = (mu[0] - mu[1]) / jnp.sqrt(hparams['two_beta2'] + s2_pred[0] + s2_pred[1])
        s2 = s2 + dt[:, 1] * hparams['tau2']
        combined_sigma2 = hparams['two_beta2'] + s2[0] + s2[1]
        combined_dev = jnp.sqrt(combined_sigma2)
        norm_diff = (mu[0] - mu[1]) / combined_dev
        return mu, s2, s2_pred, pred_norm_diff, combined_sigma2, combined_dev, norm_diff, outcome, matchup

    @staticmethod
    def _apply(state, matchup, mu, s2, delta, eta, hparams):
        mus, sigma2s = state
        mu_new = mu + delta * jnp.array([1.0, -1.0], dtype=mu.dtype)
        s2_new = s2 * jnp.maximum(1.0 - eta, hparams['kappa'])
        return (mus.at[matchup].set(mu_new), sigma2s.at[matchup].set(s2_new))

    @staticmethod
    def _step_core_bt(state, x, hparams):
        mu, s2, s2_pred, pred_nd, comb_s2, comb_dev, nd, outcome, matchup = WengLin._gather_and_inflate(
            state, x, hparams
        )
        prob = jax.nn.sigmoid(pred_nd)  # matches v1 predict() for model='bt'
        update_prob = jax.nn.sigmoid(nd)
        delta = (s2 / comb_dev) * (outcome - update_prob)
        eta = (jnp.sqrt(s2) / comb_dev) * (s2 / comb_s2) * (update_prob * (1.0 - update_prob))
        return WengLin._apply(state, matchup, mu, s2, delta, eta, hparams), prob, mu, s2_pred

    @staticmethod
    def _step_core_tm(state, x, hparams):
        mu, s2, s2_pred, pred_nd, comb_s2, comb_dev, nd, outcome, matchup = WengLin._gather_and_inflate(
            state, x, hparams
        )
        eps_c = hparams['epsilon'] / comb_dev

        # win=1.0, loss=-1.0; a draw's update direction lives in v_draw, so its
        # multiplier is 1.0 (P3.2 fix — v1 scaled draw updates by the 0.5 outcome
        # with a sign-flipped v_draw, as in v1 TrueSkill)
        sign_multiplier = jnp.where(outcome == 0.0, -1.0, 1.0)
        # float32 guard floors, as in v2 TrueSkill (no-ops in float64)
        tiny = float(jnp.finfo(mu.dtype).tiny)
        win_guard = max(2.222758749e-162, tiny)
        draw_guard = max(1e-50, tiny)

        diff_win = nd * sign_multiplier - eps_c
        sign_t = jnp.where(nd < 0.0, -1.0, 1.0)
        abs_t = jnp.abs(nd)
        diff_a = eps_c - abs_t
        diff_b = -eps_c - abs_t

        # one fused evaluation of every normal cdf/pdf this step needs:
        # the prediction (matches v1 predict() for model='tm'), win and draw moments
        z = jnp.stack([pred_nd, diff_win, diff_a, diff_b])
        cdfs = norm.cdf(z)
        pdfs = norm.pdf(z)
        prob = cdfs[0]

        cdf_win = cdfs[1]
        v_win = jnp.where(cdf_win > win_guard, pdfs[1] / jnp.maximum(cdf_win, tiny), -diff_win)
        w_win = v_win * (v_win + diff_win)

        # draw moments, reference TrueSkill formulas: v_draw is odd in nd
        # (the leader loses rating on a draw) and w_draw is positive
        pdf_a, pdf_b = pdfs[2], pdfs[3]
        shared_denom = cdfs[2] - cdfs[3]
        safe_denom = jnp.maximum(shared_denom, tiny)
        v_draw = jnp.where(shared_denom < 1e-5, -nd + sign_t * eps_c, sign_t * (pdf_b - pdf_a) / safe_denom)
        w_num = (diff_a * pdf_a) - (diff_b * pdf_b)
        w_draw = jnp.where(shared_denom < draw_guard, 1.0, (w_num / safe_denom) + jnp.square(v_draw))

        is_draw = outcome == 0.5
        v = jnp.where(is_draw, v_draw, v_win)
        # w is a variance-reduction fraction, theoretically in (0, 1); float32 backstop
        w = jnp.clip(jnp.where(is_draw, w_draw, w_win), 0.0, 1.0)

        delta = sign_multiplier * (s2 / comb_dev) * v
        eta = (jnp.sqrt(s2) / comb_dev) * (s2 / comb_s2) * w
        return WengLin._apply(state, matchup, mu, s2, delta, eta, hparams), prob, mu, s2_pred

    @staticmethod
    def _step_bt(state, x, hparams):
        new_state, prob, _, _ = WengLin._step_core_bt(state, x, hparams)
        return new_state, prob

    @staticmethod
    def _step_tm(state, x, hparams):
        new_state, prob, _, _ = WengLin._step_core_tm(state, x, hparams)
        return new_state, prob

    @staticmethod
    def _step_period_start_bt(state, x, hparams):
        new_state, _, mu, s2_pred = WengLin._step_core_bt(state, x, hparams)
        return new_state, (mu, s2_pred)

    @staticmethod
    def _step_period_start_tm(state, x, hparams):
        new_state, _, mu, s2_pred = WengLin._step_core_tm(state, x, hparams)
        return new_state, (mu, s2_pred)

    def _gathered_probs(self, gathered):
        mu, s2 = gathered  # (n, 2) pre-match mus and (inflated) sigma^2
        nd = (mu[:, 0] - mu[:, 1]) / jnp.sqrt(self.hparams['two_beta2'] + s2[:, 0] + s2[:, 1])
        return jax.nn.sigmoid(nd) if self.model == 'bt' else norm.cdf(nd)

    # ------------------------------------------------------------ batched mode

    def _initial_state_batched(self):
        # the batched carry additionally tracks who has ever played, since idle
        # inflation is applied eagerly per period rather than lazily per match
        return (*self._initial_state(), jnp.zeros(self.num_competitors, dtype=bool))

    @staticmethod
    def _batched_core(state, x, hparams):
        """One padded rating period, mirroring v1 WengLin.batched_update.

        The pooled update is model-independent in v1 (it always uses the
        sigmoid probability, even for model='tm'); only the returned
        predictions differ per model, so this emits the period-start
        norm-diffs for the wrappers to turn into probabilities.
        """
        mus, sigma2s, has_played = state
        matchups, outcomes, mask, time_delta = x
        flat_idx = matchups.reshape(-1)
        mask2 = mask[:, None]

        # returned predictions come from period-start, pre-inflation state
        # (v1 calls predict() before batched_update)
        mu = mus[matchups]  # (p, 2)
        s2_pre = sigma2s[matchups]
        pred_norm_diffs = (mu[:, 0] - mu[:, 1]) / jnp.sqrt(hparams['two_beta2'] + s2_pre[:, 0] + s2_pre[:, 1])

        # brand-new actives get the current period's single unit of inflation, not the global gap
        active = jnp.zeros_like(mus).at[flat_idx].max(jnp.repeat(mask, 2)) > 0.0
        dt_eff = jnp.where(has_played, time_delta, jnp.minimum(time_delta, 1.0))
        has_played = has_played | active
        sigma2s = jnp.where(has_played, sigma2s + dt_eff * hparams['tau2'], sigma2s)

        s2 = sigma2s[matchups]
        combined_sigma2 = hparams['two_beta2'] + s2[:, 0] + s2[:, 1]
        combined_dev = jnp.sqrt(combined_sigma2)
        probs = jax.nn.sigmoid((mu[:, 0] - mu[:, 1]) / combined_dev)

        mu_updates = (s2 / combined_dev[:, None]) * (outcomes - probs)[:, None]
        mu_updates = mu_updates * jnp.array([1.0, -1.0], dtype=mus.dtype)
        gammas = jnp.sqrt(s2) / combined_dev[:, None]
        etas = gammas * (s2 / combined_sigma2[:, None]) * (probs * (1.0 - probs))[:, None]
        multipliers = jnp.maximum(1.0 - etas, hparams['kappa'])

        mu_pooled = jnp.zeros_like(mus).at[flat_idx].add((mu_updates * mask2).reshape(-1))
        neg_inf = -jnp.inf
        mult_pooled = (
            jnp.full_like(mus, neg_inf).at[flat_idx].max(jnp.where(mask2 > 0.0, multipliers, neg_inf).reshape(-1))
        )

        mus = mus + jnp.where(active, mu_pooled, 0.0)
        sigma2s = jnp.where(active, sigma2s * mult_pooled, sigma2s)
        return (mus, sigma2s, has_played), pred_norm_diffs

    @staticmethod
    def _batched_step_bt(state, x, hparams):
        new_state, pred_norm_diffs = WengLin._batched_core(state, x, hparams)
        return new_state, jax.nn.sigmoid(pred_norm_diffs)

    @staticmethod
    def _batched_step_tm(state, x, hparams):
        new_state, pred_norm_diffs = WengLin._batched_core(state, x, hparams)
        return new_state, norm.cdf(pred_norm_diffs)

    def _apply_final_inflation(self, state, final_dt):
        mus, sigma2s = state
        return (mus, sigma2s + final_dt * self.hparams['tau2'])

    @property
    def mus(self):
        return np.asarray(self.state[0])

    @property
    def sigma2s(self):
        return np.asarray(self.state[1])

    def leaderboard_ratings(self):
        return self.mus - (3.0 * np.sqrt(self.sigma2s))
