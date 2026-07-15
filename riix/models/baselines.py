"""Baseline rating systems rating system.

`mode` selects the strength statistic like v1: 'win_rate', 'wins',
'appearances', or 'random'. Count updates are order-independent, so v1's
online and batched updates are one and the same per-period pooled add; here
the count modes follow the standard v2 pattern instead (per-match updates in
the online scan, pooled adds in batched mode) — final counts are identical
sums either way, and predict_mode='period_start' / update_method='batched'
reproduce v1's block predictions exactly.

The 'random' mode replaces v1's stateful numpy rng with a jax PRNG key carried
through the scan (deterministic per seed, but a different bit stream than v1:
count state still matches v1 exactly, probabilities cannot). Its predictions
are fresh uniforms per match in every mode, like v1's predict().
"""

import numpy as np
import jax
import jax.numpy as jnp

from riix.core.base import OnlineRatingSystem, _interleave_ratings

_RATING_FNS = {
    'win_rate': lambda wins, apps: wins / apps,
    'wins': lambda wins, apps: wins,
    'appearances': lambda wins, apps: apps,
}


def _count_step_fns(rating_fn):
    """Build the scan step functions for one count-statistic mode."""

    def step_core(state, x, hparams):
        wins, apps = state
        matchup, outcome = x
        w = wins[matchup]
        a = apps[matchup]
        r = rating_fn(w, a)
        prob = jax.nn.sigmoid(hparams['temperature'] * (r[0] - r[1]))
        wins = wins.at[matchup].add(jnp.stack([outcome, 1.0 - outcome]))
        apps = apps.at[matchup].add(1.0)
        return (wins, apps), prob, w, a

    def step(state, x, hparams):
        new_state, prob, _, _ = step_core(state, x, hparams)
        return new_state, prob

    def step_period_start(state, x, hparams):
        new_state, _, w, a = step_core(state, x, hparams)
        return new_state, (w, a)

    def batched_step(state, x, hparams):
        wins, apps = state
        matchups, outcomes, mask, _time_delta = x
        w = wins[matchups]  # (p, 2)
        a = apps[matchups]
        r = rating_fn(w, a)
        probs = jax.nn.sigmoid(hparams['temperature'] * (r[:, 0] - r[:, 1]))
        flat_idx = matchups.reshape(-1)
        win_contribs = jnp.stack([outcomes, 1.0 - outcomes], axis=1) * mask[:, None]
        wins = wins.at[flat_idx].add(win_contribs.reshape(-1))
        apps = apps.at[flat_idx].add(jnp.repeat(mask, 2))
        return (wins, apps), probs

    return step, step_period_start, batched_step


_STEP_FNS = {mode: _count_step_fns(fn) for mode, fn in _RATING_FNS.items()}


class BaselineRatingSystem(OnlineRatingSystem):
    """Counting/random baselines."""

    rating_dim = 1
    _predict_state_slots = (0, 1)  # (wins, appearances)

    def __init__(
        self,
        competitors: list,
        mode: str,  # 'win_rate', 'wins', 'appearances', or 'random'
        temperature: float = 1.0,
        prior: int = 2,
        seed: int = 0,
        update_method: str = 'online',
        predict_mode: str = 'prequential',
        dtype=jnp.float32,
    ):
        """
        Args:
            mode (str): 'win_rate', 'wins', 'appearances', or 'random' — the running statistic whose difference (through a sigmoid) is the win probability, or uniform random probabilities.
            temperature (float, optional): sharpness of the sigmoid over the statistic difference. Defaults to 1.0.
            prior (int, optional): virtual prior appearances for win_rate smoothing (half are counted as wins, floored). Defaults to 2.
            seed (int, optional): PRNG seed for mode='random'. Defaults to 0.
            update_method / predict_mode / dtype: see `OnlineRatingSystem.__init__`.
        """
        super().__init__(competitors, dtype=dtype, update_method=update_method, predict_mode=predict_mode)
        if mode not in ('random', *_RATING_FNS):
            raise ValueError(f'Invalid mode {mode}')
        self.mode = mode
        self.initial_wins = prior // 2  # v1 floors the virtual wins, keep it
        self.initial_appearances = prior
        self.seed = seed
        self.hparams = {'temperature': jnp.asarray(temperature, dtype=self.dtype)}
        if mode == 'random':
            self._step = BaselineRatingSystem._step_random
            self._step_period_start = BaselineRatingSystem._step_period_start_random
            self._batched_step = BaselineRatingSystem._batched_step_random
        else:
            self._step, self._step_period_start, self._batched_step = _STEP_FNS[mode]

    def _initial_state(self):
        counts = (
            jnp.full(self.num_competitors, self.initial_wins, dtype=self.dtype),
            jnp.full(self.num_competitors, self.initial_appearances, dtype=self.dtype),
        )
        if self.mode == 'random':
            return (*counts, jax.random.PRNGKey(self.seed))
        return counts

    # ------------------------------------------------------------ random mode

    @staticmethod
    def _step_random(state, x, hparams):
        wins, apps, key = state
        matchup, outcome = x
        key, subkey = jax.random.split(key)
        u = jax.random.uniform(subkey, (2,), dtype=wins.dtype)
        prob = jax.nn.sigmoid(hparams['temperature'] * (u[0] - u[1]))
        wins = wins.at[matchup].add(jnp.stack([outcome, 1.0 - outcome]))
        apps = apps.at[matchup].add(1.0)
        return (wins, apps, key), prob

    @staticmethod
    def _step_period_start_random(state, x, hparams):
        # random predictions are fresh per match, so the period-start "emission"
        # is simply the probability itself (no source gathering)
        return BaselineRatingSystem._step_random(state, x, hparams)

    @staticmethod
    def _batched_step_random(state, x, hparams):
        wins, apps, key = state
        matchups, outcomes, mask, _time_delta = x
        key, subkey = jax.random.split(key)
        u = jax.random.uniform(subkey, matchups.shape, dtype=wins.dtype)
        probs = jax.nn.sigmoid(hparams['temperature'] * (u[:, 0] - u[:, 1]))
        flat_idx = matchups.reshape(-1)
        win_contribs = jnp.stack([outcomes, 1.0 - outcomes], axis=1) * mask[:, None]
        wins = wins.at[flat_idx].add(win_contribs.reshape(-1))
        apps = apps.at[flat_idx].add(jnp.repeat(mask, 2))
        return (wins, apps, key), probs

    # -------------------------------------------------------------------------

    def _period_start_probs(self, emitted, sources):
        if self.mode == 'random':
            return emitted  # already per-match probabilities, fresh per match
        return super()._period_start_probs(emitted, sources)

    def _gathered_probs(self, gathered):
        if self.mode == 'random':
            # only reached from the prequential ratings path, whose emissions
            # are already the per-match probabilities
            return gathered[0]
        w, a = gathered  # (n, 2) pre-match wins and appearances
        r = _RATING_FNS[self.mode](w, a)
        return jax.nn.sigmoid(self.hparams['temperature'] * (r[:, 0] - r[:, 1]))

    def _gathered_ratings(self, gathered):
        if self.mode == 'random':
            raise NotImplementedError("mode='random' has no pre-match ratings (its predictions are fresh draws)")
        w, a = gathered  # (n, 2) pre-match wins and appearances
        return _interleave_ratings((_RATING_FNS[self.mode](w, a),))  # the mode's running statistic

    def predict(self, matchups: np.ndarray, time_step: int = None, set_cache: bool = False):
        """Like v1's predict(): count-based probabilities, or fresh uniform draws in random mode."""
        if self.mode != 'random':
            return super().predict(matchups, time_step=time_step, set_cache=set_cache)
        wins, apps, key = self.state if self.state is not None else self._initial_state()
        key, subkey = jax.random.split(key)
        u = jax.random.uniform(subkey, (matchups.shape[0], 2), dtype=self.dtype)
        self.state = (wins, apps, key)  # v1's rng also advances on every predict
        return np.asarray(jax.nn.sigmoid(self.hparams['temperature'] * (u[:, 0] - u[:, 1])))

    @property
    def wins(self):
        return np.asarray(self.state[0])

    @property
    def appearances(self):
        return np.asarray(self.state[1])
