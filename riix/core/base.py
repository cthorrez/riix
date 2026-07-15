"""Base classes for rating systems, and the jax scan machinery behind the online ones.

Design of `OnlineRatingSystem` (see also the model modules in `riix/models/`):
    - `fit_dataset` runs the whole dataset through a single `jax.lax.scan` whose
      step processes one match: gather the two competitors' states, predict,
      update, scatter back. The scan body is compiled once per (step function,
      dtype, dataset shape) and cached across instances, so sweeps never
      recompile.
    - Time-based variance inflation (Glicko, Glicko2, TrueSkill, ...) is
      precomputed outside jax: a vectorized numpy pre-pass emits, for each
      competitor in each match, the elapsed rating periods since that competitor
      was last inflated. The scan carry then stays a small tuple of float arrays
      with no integer bookkeeping inside the loop.
    - Hyperparameters are passed into the jitted scan as traced jax scalars, not
      python constants, so changing them does not trigger recompilation.

Prediction semantics: by default, returned pre-match probabilities are
prequential (match i is predicted from the state after match i-1). Two options
change that:

    - predict_mode='period_start' keeps sequential updates but returns block
      predictions computed from rating-period-start state — the right
      evaluation when within-period match order is not meaningful. A
      competitor's state only changes when they play, so its period-start value
      is its live value at its first appearance in the period: the scan emits
      pre-update gathered values and a vectorized post-pass gathers them per
      match (no extra scan carry).
    - update_method='batched' scans over padded rating periods and applies one
      simultaneous update per period (predictions are inherently period-start).

`return_pre_match_ratings` works in every mode: the pre-match emissions (or the
period-start gathered state in batched mode) already carry each competitor's
pre-match state, so ratings are assembled in the same vectorized post-pass as
the probabilities, in interleaved column order. `predict()` and
`get_pre_match_ratings()` evaluate the current fitted state directly, no scan.

Calling `fit_dataset` again continues from the fitted state, exactly as if the
two datasets had been fitted as one (train -> test evaluation): the model
carries the last fitted rating period and who has appeared (numpy, alongside
the jax state), so idle-time variance inflation spans the gap between the fits.
The datasets must share one absolute clock and the next fit must start strictly
after the last fitted period — the two halves of `split_pair_dataset` qualify;
independently constructed datasets re-anchored at their own first date do not.
"""

from abc import ABC
from functools import partial
from typing import NamedTuple, Optional
import numpy as np

import jax
import jax.numpy as jnp

from riix.utils import PairDataset


class FitResult(NamedTuple):
    """
    Result of fitting a rating system on a batch or dataset.

    Fields are None unless the corresponding `return_pre_match_*` flag was passed,
    so `fit_dataset(...).probs` and `probs, ratings = fit_dataset(...)` both work
    with a single, consistent return type.
    """

    probs: Optional[np.ndarray]
    ratings: Optional[np.ndarray]


class RatingSystem(ABC):
    """
    Base class shared by all rating systems, online or static.

    Attributes:
        rating_dim (int): Dimension of competitor ratings. 1 for systems like Elo where each
                          competitor has a single rating value, more for systems like TrueSkill
                          that use multiple values (e.g., mean and standard deviation).
        competitors (list): A list of competitors within the rating system.
        num_competitors (int): The number of competitors in the system.
        leaderboard_header (str): Column header printed above the scores by print_leaderboard.
    """

    rating_dim: int
    leaderboard_header: str = 'rating'

    def __init__(self, competitors):
        """
        Initializes a new instance of a rating system with a list of competitors.

        Parameters:
            competitors (list): A list of competitors to be included in the rating system. Each
                                competitor should have a structure or identifier compatible with
                                the specific rating system's requirements.
        """
        self.competitors = competitors
        self.num_competitors = len(competitors)

    def predict(
        self,
        matchups: np.ndarray,
        time_step: int = None,
        set_cache: bool = False,
    ):
        raise NotImplementedError

    def get_pre_match_ratings(self, matchups: np.ndarray, time_step: Optional[int] = None) -> np.ndarray:
        """
        Returns the ratings for competitors at the timestep of the matchups
        Useful when using pre-match ratings as features in downstream ML pipelines

        Parameters:
            matchups (np.ndarray of shape (n,2)): competitor indices
            time_step (optional int)

        Returns:
            np.ndarray of shape (n,2): ratings for specified competitors
        """
        raise NotImplementedError

    def fit_dataset(
        self,
        dataset: PairDataset,
        return_pre_match_probs: bool = False,
        return_pre_match_ratings: bool = False,
        cache: bool = False,
    ) -> FitResult:
        """Fit the rating system on a dataset."""
        raise NotImplementedError

    def leaderboard_ratings(self) -> np.ndarray:
        """
        Returns the array of shape (num_competitors,) of scores used to rank competitors
        on the leaderboard. Override per rating system (e.g. `mus - 3 * sqrt(sigma2s)`).
        """
        raise NotImplementedError

    def print_leaderboard(self, num_places: int = None):
        """
        Prints the leaderboard of the rating system, ranked by leaderboard_ratings().

        Parameters:
            num_places (int, optional): The number of top places to display.
                                        Defaults to all competitors.
        """
        scores = np.asarray(self.leaderboard_ratings())
        if num_places is None:
            num_places = self.num_competitors
        sorted_idxs = np.argsort(-scores)[:num_places]
        max_len = min(max([len(str(comp)) for comp in self.competitors] + [10]), 25)
        print(f'{"competitor": <{max_len}}\t{self.leaderboard_header}')
        for comp_idx in sorted_idxs:
            print(f'{str(self.competitors[comp_idx]): <{max_len}}\t{scores[comp_idx]:.6f}')


@partial(jax.jit, static_argnums=(0,))
def _fit_scan(step_fn, state, hparams, xs):
    """Run one jitted scan of `step_fn` over the matches (or rating periods) in `xs`."""

    def body(carry, x):
        return step_fn(carry, x, hparams)

    return jax.lax.scan(body, state, xs)


def _gather_emitted(emitted, sources):
    """
    Index each emitted (n, 2, ...) array by an (n, 2) occurrence-index map,
    yielding the tuple of (n, 2, ...) gathered values `_gathered_probs` /
    `_gathered_ratings` operate on. With the `compute_period_start_sources` map
    this turns per-match emissions into period-start values; the prequential
    paths skip it (the emissions are already each match's own pre-match values).
    """
    if not isinstance(emitted, tuple):
        emitted = (emitted,)
    return tuple(e.reshape(-1, *e.shape[2:])[sources] for e in emitted)


def _interleave_ratings(cols):
    """Stack per-competitor (n, 2) columns into the (n, 2*rating_dim) column order (all of c1's values, then c2's)."""
    return jnp.stack(cols, axis=-1).reshape(cols[0].shape[0], -1)


_BATCHED_RATINGS_STEPS = {}


def _batched_step_with_ratings(batched_step_fn, slots):
    """
    Wrap a batched step to additionally emit the period-start gathered state of
    the two competitors in each (padded) match — `get_pre_match_ratings` reads
    exactly that state, before the period's inflation and update. The duplicate
    gathers are CSE'd by XLA, and caching by (step, slots) keeps the wrapper's
    identity stable so the jitted scan compiles once per model.
    """
    key = (batched_step_fn, slots)
    if key not in _BATCHED_RATINGS_STEPS:

        def step(state, x, hparams):
            gathered = tuple(state[i][x[0]] for i in slots)
            new_state, probs = batched_step_fn(state, x, hparams)
            return new_state, (probs, *gathered)

        _BATCHED_RATINGS_STEPS[key] = step
    return _BATCHED_RATINGS_STEPS[key]


class InflationSchedule(NamedTuple):
    """Precomputed variance-inflation increments (see compute_inflation_dt)."""

    dts: np.ndarray  # (n, 2, 2): [..., 0] applied pre-prediction, [..., 1] post
    final_dt: np.ndarray  # (num_competitors,): idle inflation owed at fit end


def compute_period_start_sources(matchups: np.ndarray, time_steps: np.ndarray):
    """
    For each competitor-occurrence, the flat occurrence index (into the (2n,)
    row-major flattening of matchups) of that competitor's *first* appearance in
    the same rating period.

    A competitor's state only changes when they play, so their live pre-update
    state at that first appearance IS their period-start state — this map lets
    period-start predictions be assembled from values emitted by the ordinary
    prequential scan, with no extra scan carry (see predict_mode='period_start').
    """
    n = matchups.shape[0]
    flat_comp = matchups.reshape(-1)
    flat_ts = np.repeat(np.asarray(time_steps, dtype=np.int64), 2)
    order = np.argsort(flat_comp, kind='stable')
    comp_sorted = flat_comp[order]
    ts_sorted = flat_ts[order]

    run_start = np.empty(comp_sorted.shape, dtype=bool)
    run_start[0] = True
    run_start[1:] = (comp_sorted[1:] != comp_sorted[:-1]) | (ts_sorted[1:] != ts_sorted[:-1])
    run_start_idx = np.maximum.accumulate(np.where(run_start, np.arange(2 * n), 0))

    sources = np.empty(2 * n, dtype=np.int64)
    sources[order] = order[run_start_idx]
    return sources.reshape(n, 2)


def compute_inflation_dt(
    matchups: np.ndarray,
    time_steps: np.ndarray,
    initial_prev_time_step: int = -1,
    num_competitors: int = 0,
    has_played: np.ndarray = None,
):
    """
    Compute elapsed rating periods for each competitor in each match, replicating
    per-period `increase_rating_dev` semantics lazily (per appearance instead of
    per period). Fully vectorized numpy, no python loop over matches.

    At the start of every rating period containing matches, every competitor
    that has ever played is inflated by `time_delta = time_step -
    prev_time_step` (and capped). Between two appearances of a competitor those
    deltas telescope, and because the inflation increment is non-negative and the
    per-step cap is a running `min`, applying the summed delta lazily at the next
    appearance gives the exact same state.

    The elapsed time is split into two increments per appearance because of when
    predictions happen: pre-match probabilities are computed *before* the current
    period's inflation, but *after* every earlier period's. With `p` the previous
    period containing any matches and `t_last` the competitor's last appearance:

    - `dt_pre`  = p - t_last, applied before predicting (inflation the competitor
      accumulated while idle during earlier periods)
    - `dt_post` = t - p, applied after predicting, before the update (the current
      period's inflation)
    - repeat appearance within the same period: both 0 (inflation is once per period)
    - first appearance ever: dt_pre = 0 and dt_post = min(t - p, 1): a brand-new
      competitor receives the current rating period's single unit of dynamics
      (as in the Glicko-2 update, where every competitor playing in a period
      gets that period's inflation) but no idle history — before the P3.2 fix
      they received the full gap t - p to the previous period containing any
      matches, an arbitrary global quantity unrelated to the competitor.

    Args:
        matchups: (n, 2) int array of competitor indices, in chronological order.
        time_steps: (n,) int array of the rating period of each match.
        initial_prev_time_step: the model's `prev_time_step` at construction
            (-1 for Glicko/Glicko2, 0 for TrueSkill), or, when continuing from an
            earlier `fit_dataset` call, the last rating period fitted so far.
        num_competitors: length of the returned final_dt array.
        has_played: (num_competitors,) bool array of who appeared in earlier fits,
            for continuation. A previous fit's final inflation left every such
            competitor inflated exactly through `initial_prev_time_step`, so their
            first appearance here resumes idle inflation from that period instead
            of being treated as brand new, and the ones absent from this dataset
            still owe idle inflation for the full span of it.

    Returns:
        An InflationSchedule of float64 arrays:
        dts: (n, 2, 2) aligned with matchups: [..., 0] is dt_pre and [..., 1] is
            dt_post.
        final_dt: (num_competitors,) periods elapsed between each competitor's
            last appearance and the final rating period (0 for competitors who
            never appear). Idle competitors keep inflating every period, so this
            must be applied once after the scan.
    """
    n = matchups.shape[0]
    if n == 0:
        return InflationSchedule(np.zeros((0, 2, 2)), np.zeros(num_competitors))

    flat_comp = matchups.reshape(-1)  # (2n,)
    flat_ts = np.repeat(np.asarray(time_steps, dtype=np.int64), 2)  # (2n,)

    # stable sort by competitor keeps each competitor's appearances in match order
    order = np.argsort(flat_comp, kind='stable')
    comp_sorted = flat_comp[order]
    ts_sorted = flat_ts[order]

    last_ts = np.empty_like(ts_sorted)
    last_ts[1:] = ts_sorted[:-1]
    last_ts[0] = 0
    is_first = np.empty(comp_sorted.shape, dtype=bool)
    is_first[0] = True
    is_first[1:] = comp_sorted[1:] != comp_sorted[:-1]  # group boundary = first appearance

    # previous period containing any matches, for each occurrence's period
    unique_ts = np.unique(time_steps)
    period_idx = np.searchsorted(unique_ts, ts_sorted)
    prev_period = np.where(period_idx > 0, unique_ts[np.maximum(period_idx - 1, 0)], initial_prev_time_step)

    last_ts[is_first] = prev_period[is_first]  # first appearance: no pre-prediction inflation
    is_new = is_first.copy()
    if has_played is not None:
        # continuation: competitors from earlier fits resume from initial_prev_time_step
        is_new &= ~has_played[comp_sorted]
        last_ts[is_first & has_played[comp_sorted]] = initial_prev_time_step
    dt_pre = np.maximum(prev_period - last_ts, 0)
    dt_post = ts_sorted - np.maximum(prev_period, last_ts)
    # brand-new competitors get the current period's single unit of dynamics, not the global gap
    dt_post = np.where(is_new, np.minimum(dt_post, 1), dt_post)

    dts = np.empty((2 * n, 2))
    dts[order, 0] = dt_pre
    dts[order, 1] = dt_post

    # idle inflation still owed at fit end: last period minus last appearance
    # (for continuation, earlier fits' competitors absent from this dataset owe the whole span)
    is_last = np.empty(comp_sorted.shape, dtype=bool)
    is_last[-1] = True
    is_last[:-1] = comp_sorted[1:] != comp_sorted[:-1]
    final_dt = np.zeros(num_competitors)
    if has_played is not None:
        final_dt[has_played] = unique_ts[-1] - initial_prev_time_step
    final_dt[comp_sorted[is_last]] = unique_ts[-1] - ts_sorted[is_last]

    return InflationSchedule(dts.reshape(n, 2, 2), final_dt)


def pad_rating_periods(matchups, outcomes, time_steps, initial_prev_time_step):
    """
    Reshape a chronological match list into padded per-period arrays so a scan
    can step over rating periods instead of matches (update_method='batched').

    Returns:
        padded_matchups: (num_periods, max_len, 2) int32, zero-filled padding.
        padded_outcomes: (num_periods, max_len) float64.
        mask: (num_periods, max_len) bool, True for real matches. Flattening
            row-major and applying the mask recovers the original match order.
        time_deltas: (num_periods,) float64 gaps between consecutive periods
            (the first relative to `initial_prev_time_step`).
    """
    n = matchups.shape[0]
    unique_ts, starts = np.unique(time_steps, return_index=True)
    counts = np.diff(np.append(starts, n))
    num_periods, max_len = len(unique_ts), int(counts.max())

    period_of_match = np.repeat(np.arange(num_periods), counts)
    pos = np.arange(n) - starts[period_of_match]
    padded_matchups = np.zeros((num_periods, max_len, 2), dtype=np.int32)
    padded_outcomes = np.zeros((num_periods, max_len))
    mask = np.zeros((num_periods, max_len), dtype=bool)
    padded_matchups[period_of_match, pos] = matchups
    padded_outcomes[period_of_match, pos] = outcomes
    mask[period_of_match, pos] = True

    time_deltas = np.diff(unique_ts.astype(np.int64), prepend=np.int64(initial_prev_time_step)).astype(np.float64)
    return padded_matchups, padded_outcomes, mask, time_deltas


class OnlineRatingSystem(RatingSystem):
    """
    Base class for online (dynamic) rating systems such as Elo, Glicko, or TrueSkill,
    whose ratings evolve as matchups are processed. The whole fit is one jitted
    jax scan (see the module docstring for the design).

    Subclasses set `uses_dt` / `initial_prev_time_step` as needed and implement:
        - `_initial_state()` -> tuple of jnp arrays (the scan carry)
        - `_step(state, x, hparams)` -> (new_state, prob), a staticmethod where x
          is (matchup, outcome), plus a (2, 2) dt of (pre, post) inflation
          increments per competitor when `uses_dt` (see compute_inflation_dt)
        - `_step_period_start` (same as `_step` but emitting each match's
          pre-update gathered values instead of a probability) and
          `_gathered_probs(gathered)`, computing the `predict()` formula from
          a tuple of (n, 2, ...) gathered values structured like those emissions
          — together they power predict_mode='period_start', prequential
          `return_pre_match_ratings`, and the standalone `predict()`
    and store their hyperparameters as a dict of jnp scalars in `self.hparams`.

    Overridable capabilities:
        - `_predict_state_slots`: indices into the state tuple of the arrays the
          emissions gather from, in the same order (used by `predict()` /
          `get_pre_match_ratings()` and the batched ratings wrapper).
        - `_gathered_ratings(gathered)`: the default interleaves the first
          `rating_dim` gathered values in column order; override to convert
          representations (e.g. variance -> deviation) or to raise where
          pre-match ratings are not meaningful.
        - update_method='batched' additionally needs `_batched_step(state, x,
          hparams)` stepping over one padded rating period, x = (matchups (p, 2),
          outcomes (p,), mask (p,), time_delta), and `_initial_state_batched()`
          if the batched carry differs from the online one.
    """

    uses_dt = False
    initial_prev_time_step = -1
    _predict_state_slots = (0,)

    def __init__(self, competitors, dtype=jnp.float32, update_method='online', predict_mode='prequential'):
        """
        Args:
            competitors (list): competitor identifiers; matchup indices index into this list.
            dtype: jnp dtype for state and computation. Defaults to float32; float64 additionally
                requires `jax.config.update('jax_enable_x64', True)` before fitting.
            update_method (str): 'online' (sequential per-match updates) or 'batched' (one
                simultaneous update per rating period, on models that support it).
            predict_mode (str): 'prequential' (default — each match predicted from the state after
                the previous match) or 'period_start' (block predictions per rating period).
                Ignored when update_method='batched', which is inherently period-start.
        """
        super().__init__(competitors)
        if update_method not in ('online', 'batched'):
            raise ValueError(f"update_method must be 'online' or 'batched', got {update_method!r}")
        if predict_mode not in ('prequential', 'period_start'):
            raise ValueError(f"predict_mode must be 'prequential' or 'period_start', got {predict_mode!r}")
        self.dtype = jnp.dtype(dtype)
        self.update_method = update_method
        self.predict_mode = predict_mode  # ignored when update_method='batched' (inherently period-start)
        self.state = None  # set by fit_dataset
        # continuation bookkeeping (numpy, alongside the jax state), set by fit_dataset:
        # the last rating period fitted so far and who has ever appeared
        self._prev_time_step = None
        self._has_played = None

    def _initial_state(self):
        raise NotImplementedError

    def _initial_state_batched(self):
        return self._initial_state()

    def _set_state_slot(self, slot, values):
        """Replace one array of the state tuple (initializing the state if unfitted)."""
        if self.state is not None:
            state = list(self.state)
        else:
            state = list(self._initial_state_batched() if self.update_method == 'batched' else self._initial_state())
        state[slot] = jnp.asarray(values, dtype=self.dtype)
        self.state = tuple(state)

    @staticmethod
    def _step(state, x, hparams):
        raise NotImplementedError

    @staticmethod
    def _step_period_start(state, x, hparams):
        raise NotImplementedError

    def _gathered_probs(self, gathered):
        """Win probabilities from a tuple of (n, 2, ...) gathered values (the `predict()` formula)."""
        raise NotImplementedError

    def _gathered_ratings(self, gathered):
        """Pre-match ratings (n, 2 * rating_dim) from gathered values, in interleaved column order."""
        return _interleave_ratings(gathered[: self.rating_dim])

    def _period_start_probs(self, emitted, sources):
        """Assemble period-start block predictions from the scan's per-match emissions."""
        return self._gathered_probs(_gather_emitted(emitted, sources))

    def _predict_inputs(self):
        """The state arrays predictions gather from, structured like the per-match emissions."""
        state = self.state if self.state is not None else self._initial_state()
        return tuple(state[i] for i in self._predict_state_slots)

    def predict(self, matchups: np.ndarray, time_step: int = None, set_cache: bool = False):
        """Win probabilities for `matchups` from the current fitted state, no scan.

        `time_step` and `set_cache` are accepted for signature compatibility and ignored.
        """
        gathered = tuple(a[jnp.asarray(matchups, dtype=jnp.int32)] for a in self._predict_inputs())
        return np.asarray(self._gathered_probs(gathered))

    def get_pre_match_ratings(self, matchups: np.ndarray, time_step: int = None) -> np.ndarray:
        """Ratings (n, 2 * rating_dim) for `matchups` from the current fitted state, in interleaved column order."""
        gathered = tuple(a[jnp.asarray(matchups, dtype=jnp.int32)] for a in self._predict_inputs())
        return np.asarray(self._gathered_ratings(gathered))

    @staticmethod
    def _batched_step(state, x, hparams):
        raise NotImplementedError("this system does not support update_method='batched'")

    def _apply_final_inflation(self, state, final_args):
        """Inflate variances of idle competitors up to the final rating period (dt models only)."""
        raise NotImplementedError

    def _continuation_prev_time_step(self):
        """The `prev_time_step` this fit starts from: the class default on a fresh
        model, or the last fitted rating period when continuing."""
        return self.initial_prev_time_step if self._prev_time_step is None else self._prev_time_step

    def _check_continuation(self, dataset):
        """Continuing a dt model requires the next dataset to start strictly after the
        last fitted period, on the same absolute clock (e.g. the two halves of
        `split_pair_dataset`). A dataset re-anchored at its own first date would
        silently skip the idle-time inflation between the fits; a dataset starting
        inside the last fitted period cannot be pooled with the matches of that
        period that were already fitted."""
        if self.uses_dt and self._prev_time_step is not None and dataset.time_steps[0] <= self._prev_time_step:
            raise ValueError(
                f'continuation requires time_steps on one absolute clock, starting strictly after the last '
                f'fitted rating period ({self._prev_time_step}); got a dataset starting at {dataset.time_steps[0]}'
            )

    def _update_time_bookkeeping(self, dataset):
        if getattr(dataset, 'time_steps', None) is None:
            return
        self._prev_time_step = int(dataset.time_steps[-1])
        if self.uses_dt:
            appeared = np.zeros(self.num_competitors, dtype=bool) if self._has_played is None else self._has_played
            appeared[dataset.matchups.reshape(-1)] = True
            self._has_played = appeared

    def _prepare_online_xs(self, dataset, matchups, outcomes):
        """
        Build the online scan xs, plus the argument later passed to
        `_apply_final_inflation` (None to skip that pass).

        The default covers systems whose time dynamics are fully described by the
        standard (dt_pre, dt_post) increments; systems needing more per-appearance
        inputs (e.g. VSKF's dampened noise accumulation) override this to extend
        the xs tuple with extra leading-axis-n arrays.
        """
        if not self.uses_dt:
            return (matchups, outcomes), None
        schedule = compute_inflation_dt(
            dataset.matchups,
            dataset.time_steps,
            self._continuation_prev_time_step(),
            self.num_competitors,
            has_played=self._has_played,
        )
        step_xs = (matchups, outcomes, jnp.asarray(schedule.dts, dtype=self.dtype))
        return step_xs, jnp.asarray(schedule.final_dt, dtype=self.dtype)

    def fit_dataset(
        self,
        dataset,
        return_pre_match_probs: bool = False,
        return_pre_match_ratings: bool = False,
        cache: bool = False,
    ) -> FitResult:
        """Fit on the whole dataset in one jitted scan over matches (or rating periods).

        Args:
            dataset: a TimedPairDataset; a plain PairDataset works when the configuration needs no
                rating periods (no time dynamics, update_method='online', predict_mode='prequential').
            return_pre_match_probs (bool): include each match's pre-match win probability.
            return_pre_match_ratings (bool): include each match's pre-match ratings, in interleaved
                column order (competitor 1's values interleaved with competitor 2's).
            cache (bool): accepted for signature compatibility and ignored.

        Returns:
            FitResult: `probs` (n,) and `ratings` (n, 2 * rating_dim) numpy arrays, each None
            unless requested.

        Calling fit_dataset again continues from the fitted state exactly like one fit of the
        concatenated data (train -> test evaluation), provided the datasets share one absolute
        clock and the new data starts strictly after the last fitted rating period.
        """
        needs_periods = self.uses_dt or self.update_method == 'batched' or self.predict_mode == 'period_start'
        if needs_periods and getattr(dataset, 'time_steps', None) is None:
            raise ValueError(f'{type(self).__name__} in this configuration requires a TimedPairDataset (time_steps)')
        self._check_continuation(dataset)
        if self.update_method == 'batched':
            return self._fit_dataset_batched(dataset, return_pre_match_probs, return_pre_match_ratings)

        matchups = jnp.asarray(dataset.matchups, dtype=jnp.int32)
        outcomes = jnp.asarray(dataset.outcomes, dtype=self.dtype)
        step_xs, final_args = self._prepare_online_xs(dataset, matchups, outcomes)

        state = self.state if self.state is not None else self._initial_state()
        probs = None
        ratings = None
        if self.predict_mode == 'period_start' or return_pre_match_ratings:
            # same scan as prequential, but each step emits its pre-update gathered
            # values; probabilities and ratings are then assembled in one vectorized
            # pass — from each competitor's first appearance per period in
            # period_start mode, or directly per match in prequential mode
            state, emitted = _fit_scan(self._step_period_start, state, self.hparams, step_xs)
            if self.predict_mode == 'period_start':
                if return_pre_match_probs or return_pre_match_ratings:
                    sources = jnp.asarray(compute_period_start_sources(dataset.matchups, dataset.time_steps))
                if return_pre_match_ratings:
                    ratings = self._gathered_ratings(_gather_emitted(emitted, sources))
                if return_pre_match_probs:
                    probs = self._period_start_probs(emitted, sources)
            else:
                gathered = emitted if isinstance(emitted, tuple) else (emitted,)
                if return_pre_match_ratings:
                    ratings = self._gathered_ratings(gathered)
                if return_pre_match_probs:
                    probs = self._gathered_probs(gathered)
        else:
            state, probs = _fit_scan(self._step, state, self.hparams, step_xs)
        if final_args is not None:
            state = self._apply_final_inflation(state, final_args)
        self.state = state
        self._update_time_bookkeeping(dataset)

        return FitResult(
            probs=np.asarray(probs) if return_pre_match_probs else None,
            ratings=np.asarray(ratings) if return_pre_match_ratings else None,
        )

    def _fit_dataset_batched(self, dataset, return_pre_match_probs, return_pre_match_ratings):
        """Fit with one simultaneous update per rating period ('batched' semantics)."""
        padded_matchups, padded_outcomes, mask, time_deltas = pad_rating_periods(
            dataset.matchups, dataset.outcomes, dataset.time_steps, self._continuation_prev_time_step()
        )
        xs = (
            jnp.asarray(padded_matchups, dtype=jnp.int32),
            jnp.asarray(padded_outcomes, dtype=self.dtype),
            jnp.asarray(mask, dtype=self.dtype),
            jnp.asarray(time_deltas, dtype=self.dtype),
        )
        state = self.state if self.state is not None else self._initial_state_batched()
        ratings = None
        if return_pre_match_ratings:
            step = _batched_step_with_ratings(self._batched_step, self._predict_state_slots)
            state, (probs, *emitted) = _fit_scan(step, state, self.hparams, xs)
            ratings = self._gathered_ratings(tuple(e[mask] for e in emitted))
        else:
            state, probs = _fit_scan(self._batched_step, state, self.hparams, xs)
        self.state = state
        self._update_time_bookkeeping(dataset)

        return FitResult(
            probs=np.asarray(probs)[mask] if return_pre_match_probs else None,
            ratings=np.asarray(ratings) if return_pre_match_ratings else None,
        )


class StaticRatingSystem(RatingSystem):
    """
    Base class for static (non-dynamic) rating systems, which fit a single fixed set of
    ratings to the entire dataset at once (e.g. batch MLE Bradley-Terry, Massey) instead
    of evolving ratings match by match.
    """

    def fit(self, matchups: np.ndarray, outcomes: np.ndarray, time_steps: Optional[np.ndarray] = None):
        """
        Fit the ratings on all matchups at once.

        Parameters:
            matchups (np.ndarray): (n, 2) array of competitor indices.
            outcomes (np.ndarray): Array of outcomes represented as win (1), loss (0), or draw (0.5).
            time_steps (np.ndarray, optional): Time step of each matchup, for systems that weight by recency.
        """
        raise NotImplementedError

    def fit_dataset(
        self,
        dataset: PairDataset,
        return_pre_match_probs: bool = False,
        return_pre_match_ratings: bool = False,
        cache: bool = False,
    ) -> FitResult:
        """
        Fit on the whole dataset at once, then predict its matchups in-sample.

        Note that unlike an online system, the returned probs and ratings come from the
        ratings fit on ALL matchups, they are in-sample rather than truly "pre-match".
        The signature matches OnlineRatingSystem.fit_dataset so that riix.eval works
        with both kinds of systems.
        """
        self.fit(dataset.matchups, dataset.outcomes, time_steps=getattr(dataset, 'time_steps', None))
        probs = self.predict(dataset.matchups) if return_pre_match_probs else None
        ratings = self.get_pre_match_ratings(dataset.matchups) if return_pre_match_ratings else None
        return FitResult(probs=probs, ratings=ratings)
