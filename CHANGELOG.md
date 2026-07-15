# Changelog

## 0.1.0 (unreleased) — the jax backend

Every rating system is now implemented on a jax `jit` + `lax.scan` backend: the whole fit is one compiled scan. Warm fits are 2-30x faster than the numpy implementations they replace, and hyperparameter sweeps compile once and reuse the kernel for every configuration (hyperparameters are traced). jax is now a required dependency (previously the optional `riix[jax]` extra), which raises the minimum Python to 3.11. The `riix.v2` namespace that previewed this backend is gone — the classes live at their usual `riix.models` names, and existing code keeps working with the changes below.

### Changed

- **`fit_dataset` probabilities are prequential by default** — match *i* is predicted from the state after match *i-1*, not from rating-period-start state. Returned probabilities and metrics change for every existing user (typically: better log loss). Pass `predict_mode='period_start'` to reproduce the old behavior exactly — it remains the honest protocol when time granularity is a date and within-period order is arbitrary.
- **float32 by default** (previously float64). For float64, pass `dtype=jnp.float64` and run `jax.config.update('jax_enable_x64', True)` first.
- `BaselineRatingSystem(mode='random')` draws probabilities from a jax PRNG carried through the scan — deterministic per seed, but a different stream than the old numpy rng.
- `grid_search`/`random_search` worker pools use the spawn context (fork is unsafe once jax is initialized); for most sweeps single-process is faster anyway — sequential fits share one compiled scan while every worker recompiles.
- Calling `fit_dataset` again continues from the fitted state exactly like one fit of the concatenation (train → test evaluation), and now *validates* the timeline: datasets must share one absolute clock and start strictly after the last fitted rating period (`ValueError` otherwise).
- TrueSkill defaults to `update_method='online'` like every other system (it used to default to `'batched'`): a plain `TrueSkill()` now updates sequentially within rating periods — pass `update_method='batched'` for the old default. `variance_increase_method` ('time'/'match') is still supported, unchanged.
- **TrueSkill and WengLin draw updates fixed.** The old online draw update scaled the mu step by the 0.5 outcome and used a sign-flipped `v_draw`, so the *leader gained* rating from a draw (at half weight), and a `copysign` factor on `w_draw` zeroed the trailing player's variance update; the old batched mode used separate vector moment functions with yet another sign convention (competitor 1 always lost rating on a draw). Both modes now share the reference TrueSkill draw moments — `v_draw` is odd in the rating difference, so a draw moves the leader down and the trailer up, and `w_draw` is positive so both variances shrink — and the win moments share one set of guard constants.
- **First appearances get exactly one period of time dynamics.** A brand-new competitor used to receive the full gap since the previous rating period containing any matches (an arbitrary global quantity) as variance inflation before their first update; Glicko, Glicko2, TrueSkill, WengLin, and VSKF now give a newcomer exactly the current period's single unit of dynamics (as the Glicko-2 update prescribes for the period being played), in both update methods — a newcomer's first update no longer depends on how long the system existed before their debut. For online Glicko/Glicko2 this was already invisible (inflation is capped at the initial deviation).
- Glicko2's batched `phi_star` is now capped at the initial phi, matching the online path (it was uncapped, letting fresh competitors exceed the initial deviation on update).

### New

- `predict_mode=` on every constructor (`'prequential'` | `'period_start'`).
- `update_method='batched'` wherever it existed before, now as a padded-period scan (up to ~200x faster than the old batched python loops).
- `return_pre_match_ratings` works in every mode; standalone `predict()` and `get_pre_match_ratings()` evaluate fitted state with no scan.
- Train → test continuation across `fit_dataset` calls is exact and tested (previously undefined behavior).
- `get_pre_match_ratings` (and `return_pre_match_ratings`) implemented for `BaselineRatingSystem` count modes (the mode's running statistic), `ConstantVarianceGlicko` (ratings), and `Yuksel2024` ((rating, phi) columns like the Glicko family) — all three used to raise. `mode='random'` still raises: its predictions are fresh draws, not ratings.
- `split_pair_dataset(..., align_to_period=True)` snaps the split back to the start of the rating period containing it, so the train set holds only complete periods and train → test continuation never starts mid-period. The default remains the raw index split.
- API docs now render every model class (the old lazy registry hid them all from pdoc), and every constructor documents its hyperparameters.

### Removed

- The numpy implementations (no fallback; jax CPU wheels cover all supported platforms — see the README install note).
- `WengLinThurstoneMosteller` — use `WengLin(model='tm')`.
- `riix.v2` (its classes became the `riix.models` classes).
- Python 3.10 support (jax requires ≥3.11).

## 0.0.6 and earlier

Pre-changelog releases: numpy implementations of ~19 online rating systems with per-rating-period block predictions, online/batched update methods, and the eval/search utilities.
