"""Elo + Momentum
Invented by Clayton Thorrez (me)"""

import math
import numpy as np
import jax
import jax.numpy as jnp

from riix.core.base import OnlineRatingSystem


class EloMentum(OnlineRatingSystem):
    """Elo with momentum.

    `momentum_type` selects the update rule ('heavy_ball', 'nesterov', or 'adam')
    and with it the scan carry: heavy-ball and Nesterov carry (ratings, v), adam
    carries (ratings, mu, v, t). For adam, `momentum` is a (beta1, beta2) tuple.
    """

    rating_dim = 1

    def __init__(
        self,
        competitors: list,
        initial_rating: float = 1500.0,
        k: float = 32.0,
        alpha: float = math.log(10.0) / 400.0,
        momentum=0.2,
        momentum_type: str = 'nesterov',
        update_method: str = 'online',
        predict_mode: str = 'prequential',
        epsilon: float = 1e-8,
        dtype=jnp.float32,
    ):
        """
        Args:
            initial_rating (float, optional): initial rating for every competitor. Defaults to 1500.0.
            k (float, optional): update step size (K-factor). Defaults to 32.0.
            alpha (float, optional): rating-difference scale inside the sigmoid. Defaults to log(10)/400.
            momentum (float or tuple, optional): momentum coefficient; a (beta1, beta2) tuple for momentum_type='adam'. Defaults to 0.2.
            momentum_type (str, optional): 'heavy_ball', 'nesterov', or 'adam' — the momentum update rule (selects the scan carry too). Defaults to 'nesterov'.
            epsilon (float, optional): adam's denominator guard. Defaults to 1e-8.
            update_method / predict_mode / dtype: see `OnlineRatingSystem.__init__`.
        """
        super().__init__(competitors, dtype=dtype, update_method=update_method, predict_mode=predict_mode)
        self.initial_rating = initial_rating
        self.momentum_type = momentum_type
        self.hparams = {
            'k': jnp.asarray(k, dtype=self.dtype),
            'alpha': jnp.asarray(alpha, dtype=self.dtype),
        }
        if momentum_type == 'heavy_ball':
            self.hparams['momentum'] = jnp.asarray(momentum, dtype=self.dtype)
            self._step = EloMentum._step_heavy_ball
            self._step_period_start = EloMentum._step_period_start_heavy_ball
            self._batched_step = EloMentum._batched_step_heavy_ball
        elif momentum_type == 'nesterov':
            self.hparams['momentum'] = jnp.asarray(momentum, dtype=self.dtype)
            self._step = EloMentum._step_nesterov
            self._step_period_start = EloMentum._step_period_start_nesterov
            self._batched_step = EloMentum._batched_step_nesterov
        elif momentum_type == 'adam':
            beta1, beta2 = momentum
            self.hparams['beta1'] = jnp.asarray(beta1, dtype=self.dtype)
            self.hparams['beta2'] = jnp.asarray(beta2, dtype=self.dtype)
            self.hparams['epsilon'] = jnp.asarray(epsilon, dtype=self.dtype)
            self._step = EloMentum._step_adam
            self._step_period_start = EloMentum._step_period_start_adam
            self._batched_step = EloMentum._batched_step_adam
        else:
            raise ValueError(f'Invalid momentum_type {momentum_type}')

    def _initial_state(self):
        ratings = jnp.full(self.num_competitors, self.initial_rating, dtype=self.dtype)
        zeros = jnp.zeros(self.num_competitors, dtype=self.dtype)
        if self.momentum_type == 'adam':
            return (ratings, zeros, zeros, jnp.zeros(self.num_competitors, dtype=jnp.int32))
        return (ratings, zeros)

    # ------------------------------------------------------------- heavy ball

    @staticmethod
    def _step_core_heavy_ball(state, x, hparams):
        ratings, v = state
        matchup, outcome = x
        r = ratings[matchup]
        prob = jax.nn.sigmoid(hparams['alpha'] * (r[0] - r[1]))
        g = jnp.stack([outcome - prob, prob - outcome])
        v_new = (hparams['momentum'] * v[matchup]) + g
        v = v.at[matchup].set(v_new)
        ratings = ratings.at[matchup].add(hparams['k'] * v_new)
        return (ratings, v), prob, r

    @staticmethod
    def _step_heavy_ball(state, x, hparams):
        new_state, prob, _ = EloMentum._step_core_heavy_ball(state, x, hparams)
        return new_state, prob

    @staticmethod
    def _step_period_start_heavy_ball(state, x, hparams):
        new_state, _, r = EloMentum._step_core_heavy_ball(state, x, hparams)
        return new_state, r

    @staticmethod
    def _batched_step_heavy_ball(state, x, hparams):
        ratings, v = state
        probs, g, active = EloMentum._batched_grads(ratings, x, hparams)
        v_new = (hparams['momentum'] * v) + g
        v = jnp.where(active, v_new, v)
        ratings = jnp.where(active, ratings + hparams['k'] * v_new, ratings)
        return (ratings, v), probs

    # --------------------------------------------------------------- nesterov

    @staticmethod
    def _step_core_nesterov(state, x, hparams):
        ratings, v = state
        matchup, outcome = x
        r = ratings[matchup]
        prob = jax.nn.sigmoid(hparams['alpha'] * (r[0] - r[1]))
        g = jnp.stack([outcome - prob, prob - outcome])
        v_old = v[matchup]
        v_new = (hparams['momentum'] * v_old) + (hparams['k'] * g)
        update = (hparams['momentum'] * v_old) + ((1.0 + hparams['momentum']) * v_new)
        v = v.at[matchup].set(v_new)
        ratings = ratings.at[matchup].add(update)
        return (ratings, v), prob, r

    @staticmethod
    def _step_nesterov(state, x, hparams):
        new_state, prob, _ = EloMentum._step_core_nesterov(state, x, hparams)
        return new_state, prob

    @staticmethod
    def _step_period_start_nesterov(state, x, hparams):
        new_state, _, r = EloMentum._step_core_nesterov(state, x, hparams)
        return new_state, r

    @staticmethod
    def _batched_step_nesterov(state, x, hparams):
        ratings, v = state
        probs, g, active = EloMentum._batched_grads(ratings, x, hparams)
        v_new = (hparams['momentum'] * v) + (hparams['k'] * g)
        update = (hparams['momentum'] * v) + ((1.0 + hparams['momentum']) * v_new)
        v = jnp.where(active, v_new, v)
        ratings = jnp.where(active, ratings + update, ratings)
        return (ratings, v), probs

    # ------------------------------------------------------------------- adam

    @staticmethod
    def _adam_update(mu, v, t, g, hparams):
        """One adam step for gradient g at (already incremented) step count t."""
        mu_new = (hparams['beta1'] * mu) + ((1.0 - hparams['beta1']) * g)
        v_new = (hparams['beta2'] * v) + ((1.0 - hparams['beta2']) * (g * g))
        tf = t.astype(g.dtype)
        mu_hat = mu_new / (1.0 - (hparams['beta1'] ** tf))
        v_hat = v_new / (1.0 - (hparams['beta2'] ** tf))
        update = hparams['k'] * mu_hat / (jnp.sqrt(v_hat) + hparams['epsilon'])
        return mu_new, v_new, update

    @staticmethod
    def _step_core_adam(state, x, hparams):
        ratings, mu, v, t = state
        matchup, outcome = x
        r = ratings[matchup]
        prob = jax.nn.sigmoid(hparams['alpha'] * (r[0] - r[1]))
        g = jnp.stack([outcome - prob, prob - outcome])
        t_new = t[matchup] + 1
        mu_new, v_new, update = EloMentum._adam_update(mu[matchup], v[matchup], t_new, g, hparams)
        mu = mu.at[matchup].set(mu_new)
        v = v.at[matchup].set(v_new)
        t = t.at[matchup].set(t_new)
        ratings = ratings.at[matchup].add(update)
        return (ratings, mu, v, t), prob, r

    @staticmethod
    def _step_adam(state, x, hparams):
        new_state, prob, _ = EloMentum._step_core_adam(state, x, hparams)
        return new_state, prob

    @staticmethod
    def _step_period_start_adam(state, x, hparams):
        new_state, _, r = EloMentum._step_core_adam(state, x, hparams)
        return new_state, r

    @staticmethod
    def _batched_step_adam(state, x, hparams):
        ratings, mu, v, t = state
        probs, g, active = EloMentum._batched_grads(ratings, x, hparams)
        t_new = t + active.astype(t.dtype)
        mu_new, v_new, update = EloMentum._adam_update(mu, v, t_new, g, hparams)
        mu = jnp.where(active, mu_new, mu)
        v = jnp.where(active, v_new, v)
        t = jnp.where(active, t_new, t)
        ratings = jnp.where(active, ratings + update, ratings)
        return (ratings, mu, v, t), probs

    # ----------------------------------------------------------------- shared

    @staticmethod
    def _batched_grads(ratings, x, hparams):
        """Period-start probs, per-competitor pooled gradients, and active mask for one padded period."""
        matchups, outcomes, mask, _time_delta = x
        r = ratings[matchups]  # (p, 2)
        probs = jax.nn.sigmoid(hparams['alpha'] * (r[:, 0] - r[:, 1]))
        diff = (outcomes - probs) * mask
        flat = matchups.reshape(-1)
        g = jnp.zeros_like(ratings).at[flat].add(jnp.stack([diff, -diff], axis=1).reshape(-1))
        active = jnp.zeros_like(ratings).at[flat].max(jnp.repeat(mask, 2)) > 0.0
        return probs, g, active

    def _gathered_probs(self, gathered):
        (r,) = gathered  # (n, 2) pre-match ratings
        return jax.nn.sigmoid(self.hparams['alpha'] * (r[:, 0] - r[:, 1]))

    @property
    def ratings(self):
        return np.asarray(self.state[0])

    @property
    def v(self):
        return np.asarray(self.state[2] if self.momentum_type == 'adam' else self.state[1])

    @property
    def mu(self):
        if self.momentum_type != 'adam':
            raise AttributeError("mu only exists with momentum_type='adam'")
        return np.asarray(self.state[1])

    @property
    def t(self):
        if self.momentum_type != 'adam':
            raise AttributeError("t only exists with momentum_type='adam'")
        return np.asarray(self.state[3])

    def leaderboard_ratings(self):
        return self.ratings
