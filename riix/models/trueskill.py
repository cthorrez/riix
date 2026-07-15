"""TrueSkill rating system.

Win and draw v/w moments are both computed each step (they are scalar-cheap)
and selected by outcome, since lax.scan requires a branch-free step. Online and
batched modes share the same corrected moment formulas (P3.2: v1's online draw
updates were scaled by the 0.5 outcome with a sign-flipped v_draw, and its
batched mode used separate vector functions whose guards and draw signs
differed; both now use the reference TrueSkill draw moments, where v_draw is
odd in the rating difference so the leader loses rating on a draw).

`variance_increase_method` selects the skill dynamics like v1: 'time' adds
tau^2 per elapsed rating period through the dt pre-pass (idle competitors keep
inflating), 'match' adds tau^2 to both participants before every match with no
idle-time inflation at all.
"""

import math
import numpy as np
from scipy.stats import norm as scipy_norm
import jax.numpy as jnp
from jax.scipy.stats import norm

from riix.core.base import OnlineRatingSystem


class TrueSkill(OnlineRatingSystem):
    """TrueSkill."""

    rating_dim = 2
    leaderboard_header = 'mu - (3 * sd)'
    uses_dt = True
    initial_prev_time_step = 0  # v1 TrueSkill starts prev_time_step at 0, not -1
    _predict_state_slots = (0, 1)  # (mus, sigma^2)

    def __init__(
        self,
        competitors: list,
        initial_mu: float = 25.0,
        initial_sigma: float = 8.333,
        beta: float = 4.166,
        tau: float = 0.0833,
        draw_probability: float = 0.0,
        variance_increase_method: str = 'time',
        update_method: str = 'online',
        predict_mode: str = 'prequential',
        dtype=jnp.float32,
    ):
        """
        Args:
            initial_mu (float, optional): initial skill mean for every competitor. Defaults to 25.0.
            initial_sigma (float, optional): initial skill standard deviation. Defaults to 8.333.
            beta (float, optional): performance noise standard deviation. Defaults to 4.166.
            tau (float, optional): additive skill dynamics noise per rating period (or per match,
                see variance_increase_method). Defaults to 0.0833.
            draw_probability (float, optional): assumed probability of a draw, setting the draw margin. Defaults to 0.0.
            variance_increase_method (str, optional): 'time' adds tau^2 per elapsed rating period,
                so idle competitors keep inflating; 'match' adds tau^2 to each participant before
                every match, with no idle-time inflation. Defaults to 'time'.
            update_method / predict_mode / dtype: see `OnlineRatingSystem.__init__`.
        """
        super().__init__(competitors, dtype=dtype, update_method=update_method, predict_mode=predict_mode)
        if variance_increase_method not in ('time', 'match'):
            raise ValueError(f"variance_increase_method must be 'time' or 'match', got {variance_increase_method!r}")
        self.variance_increase_method = variance_increase_method
        self.initial_mu = initial_mu
        self.initial_sigma = initial_sigma
        epsilon = scipy_norm.ppf((draw_probability + 1.0) / 2.0) * math.sqrt(2.0) * beta
        self.hparams = {
            'two_beta2': jnp.asarray(2.0 * beta**2.0, dtype=self.dtype),
            'tau2': jnp.asarray(tau**2.0, dtype=self.dtype),
            'epsilon': jnp.asarray(epsilon, dtype=self.dtype),
        }
        if variance_increase_method == 'match':
            # per-match inflation needs no rating-period pre-pass (instance
            # override of the class attribute)
            self.uses_dt = False
            self._step = TrueSkill._step_match
            self._step_period_start = TrueSkill._step_period_start_match
            self._batched_step = TrueSkill._batched_step_match

    def _initial_state(self):
        return (
            jnp.full(self.num_competitors, self.initial_mu, dtype=self.dtype),
            jnp.full(self.num_competitors, self.initial_sigma**2.0, dtype=self.dtype),
        )

    @staticmethod
    def _finish_step(state, matchup, mu, s2_pred, s2, outcome, hparams):
        """Predict from (mu, s2_pred), update from (mu, s2): the moment math shared
        by both variance_increase_methods, after their mode-specific inflation."""
        mus, sigma2s = state

        pred_norm_diff = (mu[0] - mu[1]) / jnp.sqrt(hparams['two_beta2'] + s2_pred[0] + s2_pred[1])

        combined_sigma2 = hparams['two_beta2'] + s2[0] + s2[1]
        combined_dev = jnp.sqrt(combined_sigma2)
        norm_diff = (mu[0] - mu[1]) / combined_dev
        eps_c = hparams['epsilon'] / combined_dev

        # win=1.0, loss=-1.0; a draw's update direction lives in v_draw, so its
        # multiplier is 1.0 (P3.2 fix — v1 scaled draw updates by the 0.5 outcome)
        outcome_multiplier = jnp.where(outcome == 0.0, -1.0, 1.0)
        # v1's guard constants underflow to 0.0 in float32, which lets denormal
        # denominators through and blows up the divisions; floor them at the
        # smallest normal value of the dtype (a no-op in float64)
        tiny = float(jnp.finfo(mus.dtype).tiny)
        win_guard = max(2.222758749e-162, tiny)
        draw_guard = max(1e-50, tiny)

        t_win = norm_diff * outcome_multiplier
        diff_win = t_win - eps_c
        sign_t = jnp.where(norm_diff < 0.0, -1.0, 1.0)
        abs_t = jnp.abs(norm_diff)
        diff_a = eps_c - abs_t
        diff_b = -eps_c - abs_t

        # one fused evaluation for all the normal cdfs/pdfs this step needs:
        # the prediction (matches v1 predict()), the win moments, and the draw moments
        z = jnp.stack([pred_norm_diff, diff_win, diff_a, diff_b])
        cdfs = norm.cdf(z)
        pdfs = norm.pdf(z)
        prob = cdfs[0]

        # win/loss moments
        cdf_win = cdfs[1]
        v_win = jnp.where(cdf_win > win_guard, pdfs[1] / jnp.maximum(cdf_win, tiny), -diff_win)
        w_win = v_win * (v_win + diff_win)

        # draw moments, reference TrueSkill formulas: v_draw is odd in norm_diff
        # (the leader loses rating on a draw) and w_draw is positive
        pdf_a, pdf_b = pdfs[2], pdfs[3]
        shared_denom = cdfs[2] - cdfs[3]
        safe_denom = jnp.maximum(shared_denom, tiny)
        v_draw = jnp.where(shared_denom < 1e-5, -norm_diff + sign_t * eps_c, sign_t * (pdf_b - pdf_a) / safe_denom)
        w_num = (diff_a * pdf_a) - (diff_b * pdf_b)
        w_draw = jnp.where(shared_denom < draw_guard, 1.0, (w_num / safe_denom) + jnp.square(v_draw))

        is_draw = outcome == 0.5
        v = jnp.where(is_draw, v_draw, v_win)
        # w is a variance-reduction fraction, theoretically in (0, 1); the clip is
        # a float32 backstop that keeps sigma^2 positive under extreme inputs
        w = jnp.clip(jnp.where(is_draw, w_draw, w_win), 0.0, 1.0)

        mu_delta = (s2 / combined_dev) * v * outcome_multiplier
        mu_new = mu + mu_delta * jnp.array([1.0, -1.0], dtype=mus.dtype)
        s2_new = s2 - (jnp.square(s2) / combined_sigma2) * w

        mus = mus.at[matchup].set(mu_new)
        sigma2s = sigma2s.at[matchup].set(s2_new)
        return (mus, sigma2s), prob, mu, s2_pred

    @staticmethod
    def _step_core(state, x, hparams):
        mus, sigma2s = state
        matchup, outcome, dt = x
        mu = mus[matchup]
        s2 = sigma2s[matchup]
        # lazy variance inflation (no cap); dt[:, 0] is the part accumulated
        # before this rating period (visible to the prediction), dt[:, 1] is
        # this period's inflation (applied only for the update)
        s2 = s2 + dt[:, 0] * hparams['tau2']
        s2_pred = s2  # what a period-start prediction sees (see _gathered_probs)
        s2 = s2 + dt[:, 1] * hparams['tau2']
        return TrueSkill._finish_step(state, matchup, mu, s2_pred, s2, outcome, hparams)

    @staticmethod
    def _step_core_match(state, x, hparams):
        mus, sigma2s = state
        matchup, outcome = x
        mu = mus[matchup]
        s2_pred = sigma2s[matchup]  # predictions see the state before this match's tau^2
        s2 = s2_pred + hparams['tau2']
        return TrueSkill._finish_step(state, matchup, mu, s2_pred, s2, outcome, hparams)

    @staticmethod
    def _step(state, x, hparams):
        new_state, prob, _, _ = TrueSkill._step_core(state, x, hparams)
        return new_state, prob

    @staticmethod
    def _step_period_start(state, x, hparams):
        new_state, _, mu, s2_pred = TrueSkill._step_core(state, x, hparams)
        return new_state, (mu, s2_pred)

    @staticmethod
    def _step_match(state, x, hparams):
        new_state, prob, _, _ = TrueSkill._step_core_match(state, x, hparams)
        return new_state, prob

    @staticmethod
    def _step_period_start_match(state, x, hparams):
        new_state, _, mu, s2_pred = TrueSkill._step_core_match(state, x, hparams)
        return new_state, (mu, s2_pred)

    def _gathered_probs(self, gathered):
        mu, s2 = gathered  # (n, 2) pre-match mus and (inflated) sigma^2
        nd = (mu[:, 0] - mu[:, 1]) / jnp.sqrt(self.hparams['two_beta2'] + s2[:, 0] + s2[:, 1])
        return norm.cdf(nd)

    def _initial_state_batched(self):
        # the batched carry additionally tracks who has ever played, since idle
        # inflation is applied eagerly per period rather than lazily per match
        return (*self._initial_state(), jnp.zeros(self.num_competitors, dtype=bool))

    @staticmethod
    def _batched_finish(mus, sigma2s, has_played, matchups, outcomes, mask, mu, active, pred_norm_diffs, hparams):
        """The shared moment math and pooled update for one padded rating period,
        applied after the mode-specific variance inflation.

        Uses the same corrected win/draw moments as the online step; sigma^2
        updates are pooled by *mean* over each competitor's appearances in the
        period (v1 batched semantics).
        """
        flat_idx = matchups.reshape(-1)
        mask2 = mask[:, None]
        zeros = jnp.zeros_like(mus)
        tiny = float(jnp.finfo(mus.dtype).tiny)
        win_guard = max(2.222758749e-162, tiny)
        draw_guard = max(1e-50, tiny)

        s2 = sigma2s[matchups]
        combined_sigma2 = hparams['two_beta2'] + s2[:, 0] + s2[:, 1]
        combined_dev = jnp.sqrt(combined_sigma2)
        nd = (mu[:, 0] - mu[:, 1]) / combined_dev
        eps_c = hparams['epsilon'] / combined_dev

        outcome_multiplier = jnp.where(outcomes == 0.0, -1.0, 1.0)

        diff_win = nd * outcome_multiplier - eps_c
        sign_t = jnp.where(nd < 0.0, -1.0, 1.0)
        abs_t = jnp.abs(nd)
        diff_a = eps_c - abs_t
        diff_b = -eps_c - abs_t

        # one fused evaluation for all the normal cdfs/pdfs this period needs:
        # the returned predictions (period-start state), win and draw moments
        z = jnp.stack([pred_norm_diffs, diff_win, diff_a, diff_b])
        cdfs = norm.cdf(z)
        pdfs = norm.pdf(z)
        probs = cdfs[0]  # matches v1 predict(), called before batched_update

        cdf_win = cdfs[1]
        v_win = jnp.where(cdf_win > win_guard, pdfs[1] / jnp.maximum(cdf_win, tiny), -diff_win)
        w_win = v_win * (v_win + diff_win)

        pdf_a, pdf_b = pdfs[2], pdfs[3]
        shared_denom = cdfs[2] - cdfs[3]
        safe_denom = jnp.maximum(shared_denom, tiny)
        v_draw = jnp.where(shared_denom < 1e-5, -nd + sign_t * eps_c, sign_t * (pdf_b - pdf_a) / safe_denom)
        w_num = (diff_a * pdf_a) - (diff_b * pdf_b)
        w_draw = jnp.where(shared_denom < draw_guard, 1.0, (w_num / safe_denom) + jnp.square(v_draw))

        is_draw = outcomes == 0.5
        v = jnp.where(is_draw, v_draw, v_win)
        # w is a variance-reduction fraction, theoretically in (0, 1); float32 backstop
        w = jnp.clip(jnp.where(is_draw, w_draw, w_win), 0.0, 1.0)

        mu_updates = v[:, None] * (s2 / combined_dev[:, None])
        mu_updates = mu_updates * jnp.array([1.0, -1.0], dtype=mus.dtype) * outcome_multiplier[:, None]
        sigma2_updates = (jnp.square(s2) * w[:, None]) / combined_sigma2[:, None]

        mu_pooled = zeros.at[flat_idx].add((mu_updates * mask2).reshape(-1))
        counts = zeros.at[flat_idx].add(jnp.repeat(mask, 2))
        sigma2_pooled = zeros.at[flat_idx].add((sigma2_updates * mask2).reshape(-1)) / jnp.maximum(counts, 1.0)

        mus = mus + jnp.where(active, mu_pooled, 0.0)
        sigma2s = jnp.where(active, sigma2s - sigma2_pooled, sigma2s)
        return (mus, sigma2s, has_played), probs

    @staticmethod
    def _batched_step(state, x, hparams):
        """One padded rating period with per-period time inflation."""
        mus, sigma2s, has_played = state
        matchups, outcomes, mask, time_delta = x
        flat_idx = matchups.reshape(-1)

        mu = mus[matchups]  # (p, 2)
        s2_pre = sigma2s[matchups]
        pred_norm_diffs = (mu[:, 0] - mu[:, 1]) / jnp.sqrt(hparams['two_beta2'] + s2_pre[:, 0] + s2_pre[:, 1])

        # uncapped inflation for everyone who has ever played, active included;
        # brand-new actives get the current period's single unit, not the global gap
        active = jnp.zeros_like(mus).at[flat_idx].max(jnp.repeat(mask, 2)) > 0.0
        dt_eff = jnp.where(has_played, time_delta, jnp.minimum(time_delta, 1.0))
        has_played = has_played | active
        sigma2s = jnp.where(has_played, sigma2s + dt_eff * hparams['tau2'], sigma2s)

        return TrueSkill._batched_finish(
            mus, sigma2s, has_played, matchups, outcomes, mask, mu, active, pred_norm_diffs, hparams
        )

    @staticmethod
    def _batched_step_match(state, x, hparams):
        """One padded rating period with per-match variance increase: each
        competitor's sigma^2 grows by tau^2 times their number of matches in the
        period (v1 batched_update with variance_increase_method='match'); idle
        competitors are untouched."""
        mus, sigma2s, has_played = state
        matchups, outcomes, mask, _time_delta = x
        flat_idx = matchups.reshape(-1)

        mu = mus[matchups]  # (p, 2)
        s2_pre = sigma2s[matchups]
        pred_norm_diffs = (mu[:, 0] - mu[:, 1]) / jnp.sqrt(hparams['two_beta2'] + s2_pre[:, 0] + s2_pre[:, 1])

        counts = jnp.zeros_like(mus).at[flat_idx].add(jnp.repeat(mask, 2))
        active = counts > 0.0
        has_played = has_played | active
        sigma2s = sigma2s + counts * hparams['tau2']

        return TrueSkill._batched_finish(
            mus, sigma2s, has_played, matchups, outcomes, mask, mu, active, pred_norm_diffs, hparams
        )

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
        return self.mus - 3.0 * np.sqrt(self.sigma2s)
