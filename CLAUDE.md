# CLAUDE.md

riix is a Python library of vectorized online rating systems (Elo, Glicko, TrueSkill, Weng-Lin, and ~15 more) built for **offline experimentation on paired-comparison datasets** — comparing predictive accuracy of rating systems on historical data with a known, fixed set of competitors. It is deliberately NOT for streaming/production use with new competitors arriving over time. Matchups are strictly 1v1: a competitor can be a single player or a whole team rated as one unit, but per-player ratings within multi-player teams are deliberately out of scope — never suggest or add team support (an implementation was built and removed in July 2026).

## Commands

- `uv sync` — create/update the environment (Python version pinned in `.python-version`)
- `uv run pytest tests/ -q` — run tests
- `uv run pre-commit run --all-files` — lint (ruff), format, and verify uv.lock is in sync
- `uv build` — build sdist/wheel (uv_build backend, flat layout via `module-root = ""`)
- `uv run python benchmarks/ab.py` — speed-compare the working tree against main (see `benchmarks/README.md`); run before/after performance-motivated changes
- Docs are built ONLY by CI (`.github/workflows/docs.yml`) with pdoc into `_site/` (gitignored). `docs/` contains only hand-made pdoc templates and assets (theme.css, syntax-highlighting.css, logo, favicon). Never commit generated HTML.

## Architecture

- `riix/core/base.py` — `RatingSystem` (root), `OnlineRatingSystem` (the jax scan base), `StaticRatingSystem`, `FitResult`, and the shared scan machinery (`_fit_scan`, `compute_inflation_dt`, `pad_rating_periods`, ...). `fit_dataset(dataset, return_pre_match_probs=..., return_pre_match_ratings=...)` lowers the whole fit into one jitted `lax.scan`; step functions are staticmethods (their identity is the jit key — one compile per (model class, mode, dtype, dataset shape)); hyperparameters are traced jnp scalars so sweeps never recompile. Probabilities are prequential by default; `predict_mode='period_start'` gives per-period block predictions; `update_method='batched'` scans over padded rating periods. Calling `fit_dataset` again continues exactly like one fit of the concatenation (requires one absolute clock, strictly later start).
- `riix/models/*.py` — one rating system per file. Pattern (see `models/template.py` and `models/elo.py`): class attribute `rating_dim`, constructor takes `competitors` first plus hyperparameters (stored as traced jnp scalars in `self.hparams`), then `update_method`/`predict_mode`/`dtype`. Implement `_initial_state`, `_step`/`_step_period_start` (thin wrappers over a `_step_core` that returns `(state, prob, *emitted)`), `_gathered_probs`, and optionally `_batched_step`. State is exposed via numpy-converting properties matching historical attribute names (`ratings`, `mus`, ...).
- `riix/utils/data_utils.py` — `PairDataset` / `TimedPairDataset`. Maps competitor names to int indices via polars joins (`maintain_order='left'` is load-bearing; polars does not otherwise guarantee join row order). Input dataframes MUST be sorted chronologically — the constructor raises if not.
- `riix/metrics.py` — binary metrics; `binary_metrics_suite` is the standard evaluation bundle. `riix/eval.py` — `evaluate` plus `grid_search`, `random_search`, and `optuna_search` (grid/random are multiprocessing-capable via a spawn-context Pool — but single-process is usually faster since sequential fits share one compiled scan; optuna is an optional extra).
- The jax migration is complete (July 2026): v1 numpy deleted at the P3.1 swap, v1 quirks fixed at P3.2, packaging and release prep done at P3.3/P3.4. `MIGRATION.md` is the historical record and decision log — consult it before revisiting any migration-era decision. The only outstanding step is publishing 0.1.0 to PyPI (stamp the date on the CHANGELOG heading when it ships).

## Conventions and gotchas

- Outcomes are encoded as float: win=1.0, draw=0.5, loss=0.0. Probabilities are always P(competitor 1 wins).
- Ruff: single quotes, line length 120, google-style docstrings (rendered by pdoc).
- jax is a **core dependency** (since 0.1.0; it's why `requires-python` is ≥3.11). float32 is the default dtype; float64 needs `dtype=` plus `jax.config.update('jax_enable_x64', True)` (tests use a module-scoped fixture). optuna remains an optional extra with a guarded import in `riix/eval.py`.
- `tests/test_all_models.py` auto-discovers every `OnlineRatingSystem` subclass in `riix/models/` and smoke-tests fit+predict. A new model gets tested for free; if its constructor requires args beyond `competitors`, add them to `EXTRA_KWARGS` there.
- The README example is the de facto quickstart — if you rename public functions, update it and keep it runnable (it uses `generate_pair_data`).
- License is CC BY-NC-SA (intentionally non-commercial); don't suggest relicensing to permissive.
