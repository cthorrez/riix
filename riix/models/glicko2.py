"""Glicko 2 rating system.

phi is carried as a variance (phi^2). The volatility update's Illinois-method
root find runs as a `lax.while_loop` inside the scan step, replicating the v1
iteration exactly but with a hard iteration cap as a safety net (float32 can
stall short of the 1e-6 convergence threshold; float64 converges in a handful
of iterations, far below the cap).
"""

import numpy as np
import jax
import jax.numpy as jnp

from riix.utils.constants import THREE_OVER_PI_SQUARED
from riix.core.base import OnlineRatingSystem, _interleave_ratings

GLICKO2_SCALE = 173.7178
_MAX_SOLVER_ITERS = 100


def _sigma_prime(phi2, delta, v, sigma, tau, tau2, epsilon):
    """Solve for the new volatility, mirroring v1 Glicko2.get_sigma_prime."""
    a = jnp.log(sigma * sigma)
    delta2 = delta * delta

    def f(x):
        ex = jnp.exp(x)
        phi2_v_ex = phi2 + v + ex
        return (ex * (delta2 - phi2_v_ex)) / (2.0 * phi2_v_ex * phi2_v_ex) - ((x - a) / tau2)

    def bracket_downward(_):
        return jax.lax.while_loop(lambda b: f(b) < 0, lambda b: b - tau, a - tau)

    b_init = jax.lax.cond(delta2 > phi2 + v, lambda _: jnp.log(delta2 - phi2 - v), bracket_downward, operand=None)

    def cond(carry):
        A, B, f_A, f_B, i = carry
        return (jnp.abs(A - B) > epsilon) & (i < _MAX_SOLVER_ITERS)

    def body(carry):
        A, B, f_A, f_B, i = carry
        C = A + ((A - B) * f_A) / (f_B - f_A)
        f_C = f(C)
        crossed = (f_C * f_B) <= 0
        A = jnp.where(crossed, B, A)
        f_A = jnp.where(crossed, f_B, f_A / 2.0)
        return (A, C, f_A, f_C, i + 1)

    A, *_ = jax.lax.while_loop(cond, body, (a, b_init, f(a), f(b_init), jnp.int32(0)))
    return jnp.exp(A / 2.0)


class Glicko2(OnlineRatingSystem):
    """Mark Glickman's Glicko 2 rating system with volatility."""

    rating_dim = 2
    leaderboard_header = 'rating - (3*dev)'
    uses_dt = True
    initial_prev_time_step = -1
    _predict_state_slots = (0, 1)  # (mus, phi^2)

    def __init__(
        self,
        competitors: list,
        initial_rating: float = 1500.0,
        initial_rd: float = 350.0,
        initial_sigma: float = 0.06,
        tau: float = 0.5,
        epsilon: float = 1e-6,
        update_method: str = 'online',
        predict_mode: str = 'prequential',
        dtype=jnp.float32,
    ):
        """
        Args:
            initial_rating (float, optional): initial rating on the display (Glicko-1) scale;
                converted internally to the Glicko-2 scale mu. Defaults to 1500.0.
            initial_rd (float, optional): initial rating deviation on the display scale (also the
                cap deviations inflate back toward while idle). Defaults to 350.0.
            initial_sigma (float, optional): initial volatility, governing how fast RD can grow
                on surprising results. Defaults to 0.06.
            tau (float, optional): constrains volatility change per update; smaller = more stable
                volatility. Defaults to 0.5.
            epsilon (float, optional): convergence tolerance of the volatility solver. Defaults to 1e-6.
            update_method / predict_mode / dtype: see `OnlineRatingSystem.__init__`.
        """
        super().__init__(competitors, dtype=dtype, update_method=update_method, predict_mode=predict_mode)
        self.initial_mu = (initial_rating - 1500.0) / GLICKO2_SCALE
        self.initial_phi = initial_rd / GLICKO2_SCALE
        self.initial_sigma = initial_sigma
        self.hparams = {
            'max_phi2': jnp.asarray(self.initial_phi**2.0, dtype=self.dtype),
            'tau': jnp.asarray(tau, dtype=self.dtype),
            'tau2': jnp.asarray(tau**2.0, dtype=self.dtype),
            'epsilon': jnp.asarray(epsilon, dtype=self.dtype),
            't3p2': jnp.asarray(THREE_OVER_PI_SQUARED, dtype=self.dtype),
        }

    def _initial_state(self):
        return (
            jnp.full(self.num_competitors, self.initial_mu, dtype=self.dtype),
            jnp.full(self.num_competitors, self.initial_phi**2.0, dtype=self.dtype),  # phi^2
            jnp.full(self.num_competitors, self.initial_sigma, dtype=self.dtype),
        )

    @staticmethod
    def _step_core(state, x, hparams):
        mus, phi2s, sigmas = state
        matchup, outcome, dt = x
        sign_flip = jnp.array([1.0, -1.0], dtype=mus.dtype)

        mu = mus[matchup]
        phi2 = phi2s[matchup]
        sigma = sigmas[matchup]

        # lazy phi inflation, capped at the initial phi; dt[:, 0] is the part
        # accumulated before this rating period (visible to the prediction),
        # dt[:, 1] is this period's inflation (applied only for the update)
        phi2 = jnp.minimum(phi2 + dt[:, 0] * jnp.square(sigma), hparams['max_phi2'])
        phi2_pred = phi2  # what a period-start prediction sees (see _gathered_probs)

        # matches v1 predict(): sigmoid(g(sqrt(phi1^2 + phi2^2)) * mu_diff)
        g_comb = 1.0 / jnp.sqrt(1.0 + (phi2[0] + phi2[1]) * hparams['t3p2'])
        prob = jax.nn.sigmoid(g_comb * (mu[0] - mu[1]))

        phi2 = jnp.minimum(phi2 + dt[:, 1] * jnp.square(sigma), hparams['max_phi2'])

        g_opp = jnp.flip(1.0 / jnp.sqrt(1.0 + phi2 * hparams['t3p2']))
        diffs = (mu * sign_flip).sum() * sign_flip  # (mu1 - mu2, mu2 - mu1)
        expected = jax.nn.sigmoid(g_opp * diffs)
        v = 1.0 / (jnp.square(g_opp) * expected * (1.0 - expected))

        outcomes_both = jnp.stack([outcome, 1.0 - outcome])
        score_diff = outcomes_both - expected
        delta = v * g_opp * score_diff

        tau, tau2, eps = hparams['tau'], hparams['tau2'], hparams['epsilon']
        sigma_new = jnp.stack(
            [
                _sigma_prime(phi2[0], delta[0], v[0], sigma[0], tau, tau2, eps),
                _sigma_prime(phi2[1], delta[1], v[1], sigma[1], tau, tau2, eps),
            ]
        )

        # phi was already inflated for the passage of time above
        phi2_new = 1.0 / ((1.0 / phi2) + (1.0 / v))
        mu_new = mu + phi2_new * g_opp * score_diff

        mus = mus.at[matchup].set(mu_new)
        phi2s = phi2s.at[matchup].set(phi2_new)
        sigmas = sigmas.at[matchup].set(sigma_new)
        return (mus, phi2s, sigmas), prob, mu, phi2_pred

    @staticmethod
    def _step(state, x, hparams):
        new_state, prob, _, _ = Glicko2._step_core(state, x, hparams)
        return new_state, prob

    @staticmethod
    def _step_period_start(state, x, hparams):
        new_state, _, mu, phi2_pred = Glicko2._step_core(state, x, hparams)
        return new_state, (mu, phi2_pred)

    def _gathered_probs(self, gathered):
        mu, phi2 = gathered  # (n, 2) pre-match mus and (inflated, capped) phi^2
        g_comb = 1.0 / jnp.sqrt(1.0 + (phi2[:, 0] + phi2[:, 1]) * self.hparams['t3p2'])
        return jax.nn.sigmoid(g_comb * (mu[:, 0] - mu[:, 1]))

    def _gathered_ratings(self, gathered):
        mu, phi2 = gathered
        return _interleave_ratings((mu, jnp.sqrt(phi2)))  # v1 columns are (mu, phi)

    def _initial_state_batched(self):
        # the batched carry additionally tracks who has ever played, since idle
        # inflation is applied eagerly per period rather than lazily per match
        return (*self._initial_state(), jnp.zeros(self.num_competitors, dtype=bool))

    @staticmethod
    def _batched_step(state, x, hparams):
        """One padded rating period, mirroring v1 Glicko2.batched_update."""
        mus, phi2s, sigmas, has_played = state
        matchups, outcomes, mask, time_delta = x
        flat_idx = matchups.reshape(-1)
        zeros = jnp.zeros(mus.shape[0], dtype=mus.dtype)

        mu = mus[matchups]  # (p, 2)
        phi2 = phi2s[matchups]

        # returned predictions, from period-start state (matches v1 predict())
        g_comb = 1.0 / jnp.sqrt(1.0 + (phi2[:, 0] + phi2[:, 1]) * hparams['t3p2'])
        probs = jax.nn.sigmoid(g_comb * (mu[:, 0] - mu[:, 1]))

        g_opp = (1.0 / jnp.sqrt(1.0 + phi2 * hparams['t3p2']))[:, ::-1]
        mu_diffs = mu - mu[:, ::-1]
        expected = jax.nn.sigmoid(g_opp * mu_diffs)
        outcomes_both = jnp.stack([outcomes, 1.0 - outcomes], axis=1)
        mask2 = mask[:, None]

        counts = zeros.at[flat_idx].add(jnp.repeat(mask, 2))
        d2_inv_sum = zeros.at[flat_idx].add((jnp.square(g_opp) * expected * (1.0 - expected) * mask2).reshape(-1))
        grads = zeros.at[flat_idx].add((g_opp * (outcomes_both - expected) * mask2).reshape(-1))
        active = counts > 0

        v = 1.0 / jnp.where(active, d2_inv_sum, 1.0)
        delta = v * grads  # 0 for inactive competitors since their grads are 0

        tau, tau2, eps = hparams['tau'], hparams['tau2'], hparams['epsilon']
        sigma_solved = jax.vmap(_sigma_prime, in_axes=(0, 0, 0, 0, None, None, None))(
            phi2s, delta, v, sigmas, tau, tau2, eps
        )
        sigmas = jnp.where(active, sigma_solved, sigmas)

        # active: the period's inflation enters through phi_star with the NEW
        # sigma, capped at the initial phi like the online path (P3.2 fix; v1
        # left phi_star uncapped); idle-but-played: capped inflation with the
        # old sigma. Brand-new actives get the current period's single unit of
        # inflation, not the global gap since the previous period with matches.
        dt_eff = jnp.where(has_played, time_delta, jnp.minimum(time_delta, 1.0))
        phi_star2 = jnp.minimum(phi2s + dt_eff * jnp.square(sigmas), hparams['max_phi2'])
        phi2_active = 1.0 / ((1.0 / phi_star2) + (1.0 / v))
        phi2_idle = jnp.minimum(phi2s + time_delta * jnp.square(sigmas), hparams['max_phi2'])
        phi2s = jnp.where(active, phi2_active, jnp.where(has_played, phi2_idle, phi2s))

        mus = mus + jnp.where(active, phi2_active * grads, 0.0)
        has_played = has_played | active
        return (mus, phi2s, sigmas, has_played), probs

    def _apply_final_inflation(self, state, final_dt):
        mus, phi2s, sigmas = state
        phi2s = jnp.minimum(phi2s + final_dt * jnp.square(sigmas), self.hparams['max_phi2'])
        return (mus, phi2s, sigmas)

    @property
    def mus(self):
        state = self.state if self.state is not None else self._initial_state()
        return np.asarray(state[0])

    @mus.setter
    def mus(self, values):
        self._set_state_slot(0, values)

    @property
    def phis(self):
        state = self.state if self.state is not None else self._initial_state()
        return np.sqrt(np.asarray(state[1]))

    @phis.setter
    def phis(self, values):
        self._set_state_slot(1, np.square(np.asarray(values)))  # carried as phi^2

    @property
    def sigmas(self):
        state = self.state if self.state is not None else self._initial_state()
        return np.asarray(state[2])

    def leaderboard_ratings(self):
        # matches v1 Glicko2.leaderboard_ratings (which uses squared phis)
        return self.mus - 3.0 * np.asarray(self.state[1])
