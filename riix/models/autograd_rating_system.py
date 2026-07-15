"""Online gradient ascent on arbitrary differentiable likelihoods rating system.

The `cdf` hyperparameter is a *callable*, so unlike the scalar hyperparameters
(traced, never recompiled) it has to be baked into the scan step function and
thereby into the static jit key: each distinct cdf object compiles its own scan.
Pass the same function object across instances (as the default `logistic.cdf`
is) and sweeps over the scalar hyperparameters still reuse one compilation.
"""

import math
import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.stats import logistic

from riix.core.base import OnlineRatingSystem

_STEP_FNS = {}


def _step_fns_for(cdf):
    """Build (or fetch cached) scan step functions closing over `cdf`.

    Caching by the cdf object keeps the step functions' identities stable, so
    the jitted scan in the base class compiles once per distinct cdf rather
    than once per model instance.
    """
    if cdf in _STEP_FNS:
        return _STEP_FNS[cdf]

    def nll(matchup_ratings, outcome, scale):
        # differentiating w.r.t. only the gathered ratings keeps the gradient
        # O(1) per match instead of O(num_competitors); the prob rides along as
        # aux so the cdf is evaluated once per step, not once more to predict
        prob = cdf((matchup_ratings[0] - matchup_ratings[1]) / scale)
        return -jnp.log((outcome * prob) + ((1.0 - outcome) * (1.0 - prob))), prob

    grad_fn = jax.grad(nll, has_aux=True)

    def nll_batch(matchup_ratings, outcomes, mask, scale):
        probs = cdf((matchup_ratings[:, 0] - matchup_ratings[:, 1]) / scale)
        return -(mask * jnp.log((outcomes * probs) + ((1.0 - outcomes) * (1.0 - probs)))).sum()

    batch_grad_fn = jax.grad(nll_batch)

    def step_core(state, x, hparams):
        (ratings,) = state
        matchup, outcome = x
        r = ratings[matchup]
        g, prob = grad_fn(r, outcome, hparams['scale'])
        update = g * hparams['lr']
        return (ratings.at[matchup].add(-update),), prob, r

    def step(state, x, hparams):
        new_state, prob, _ = step_core(state, x, hparams)
        return new_state, prob

    def step_period_start(state, x, hparams):
        new_state, _, r = step_core(state, x, hparams)
        return new_state, r

    def batched_step(state, x, hparams):
        (ratings,) = state
        matchups, outcomes, mask, _time_delta = x
        r = ratings[matchups]  # (p, 2)
        probs = cdf((r[:, 0] - r[:, 1]) / hparams['scale'])
        # masking inside the summed likelihood zeroes the padding rows' gradients
        update = batch_grad_fn(r, outcomes, mask, hparams['scale']) * hparams['lr']
        return (ratings.at[matchups.reshape(-1)].add(-update.reshape(-1)),), probs

    _STEP_FNS[cdf] = (step, step_period_start, batched_step)
    return _STEP_FNS[cdf]


class AutogradRatingSystem(OnlineRatingSystem):
    """Gradient updates on a differentiable likelihood."""

    rating_dim = 1

    def __init__(
        self,
        competitors: list,
        cdf: callable = logistic.cdf,
        scale: float = 400.0 / math.log(10.0),
        learning_rate: float = 32.0,
        initial_rating: float = 1500.0,
        update_method: str = 'online',
        predict_mode: str = 'prequential',
        dtype=jnp.float32,
    ):
        """
        Args:
            cdf (callable, optional): differentiable CDF mapping scaled rating difference to win probability. Defaults to the base-10 logistic. Its identity is the jit compilation key — share one function object across instances to reuse the compiled scan.
            scale (float, optional): divisor applied to the rating difference before the cdf (Elo-style scale). Defaults to 400/ln(10).
            learning_rate (float, optional): gradient-ascent step size on the per-match log likelihood. Defaults to 32.0.
            initial_rating (float, optional): initial rating for every competitor. Defaults to 1500.0.
            update_method / predict_mode / dtype: see `OnlineRatingSystem.__init__`.
        """
        super().__init__(competitors, dtype=dtype, update_method=update_method, predict_mode=predict_mode)
        self.initial_rating = initial_rating
        self.cdf = cdf
        self.hparams = {
            'scale': jnp.asarray(scale, dtype=self.dtype),
            'lr': jnp.asarray(learning_rate * scale, dtype=self.dtype),
        }
        self._step, self._step_period_start, self._batched_step = _step_fns_for(cdf)

    def _initial_state(self):
        return (jnp.full(self.num_competitors, self.initial_rating, dtype=self.dtype),)

    def _gathered_probs(self, gathered):
        (r,) = gathered  # (n, 2) pre-match ratings
        return self.cdf((r[:, 0] - r[:, 1]) / self.hparams['scale'])

    @property
    def ratings(self):
        return np.asarray(self.state[0])

    def leaderboard_ratings(self):
        return self.ratings
