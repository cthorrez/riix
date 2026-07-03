# CLAUDE.md

riix is a Python library of vectorized online rating systems (Elo, Glicko, TrueSkill, Weng-Lin, and ~15 more) built for **offline experimentation on paired-comparison datasets** — comparing predictive accuracy of rating systems on historical data with a known, fixed set of competitors. It is deliberately NOT for streaming/production use with new competitors arriving over time. Only 1v1 matchups are supported (two-team support planned, >2 teams never).

## Commands

- `uv sync` — create/update the environment (Python version pinned in `.python-version`)
- `uv run pytest tests/ -q` — run tests
- `uv run pre-commit run --all-files` — lint (ruff), format, and verify uv.lock is in sync
- `uv build` — build sdist/wheel (uv_build backend, flat layout via `module-root = ""`)
- `uv run python benchmarks/ab.py` — speed-compare the working tree against main (see `benchmarks/README.md`); run before/after performance-motivated changes
- Docs are built ONLY by CI (`.github/workflows/docs.yml`) with pdoc into `_site/` (gitignored). `docs/` contains only hand-made pdoc templates and assets (theme.css, syntax-highlighting.css, logo, favicon). Never commit generated HTML.

## Architecture

- `riix/core/base.py` — `RatingSystem` (base) and `OnlineRatingSystem`. The key entry point is `fit_dataset(dataset, return_pre_match_probs=..., return_pre_match_ratings=...)`, which iterates rating periods and calls `fit_batch` → `predict` + `update` per period. An inheritance split for non-dynamic (static) rating systems is in progress.
- `riix/models/*.py` — one rating system per file. Pattern (see `models/template.py` and `models/elo.py`): class attribute `rating_dim`, constructor takes `competitors` first plus hyperparameters, and an `update_method` arg ('online' or 'batched') that binds `self.update` to `online_update` or `batched_update`. `predict` returns win probabilities for `matchups` (an (n,2) int index array).
- `riix/utils/data_utils.py` — `PairDataset` / `TimedPairDataset`. Maps competitor names to int indices via polars joins (`maintain_order='left'` is load-bearing; polars does not otherwise guarantee join row order). Iterating a `TimedPairDataset` yields `(matchups, outcomes, time_step)` per rating period. Input dataframes MUST be sorted chronologically — the constructor raises if not.
- `riix/metrics.py` — binary metrics; `binary_metrics_suite` is the standard evaluation bundle. `riix/eval.py` — `evaluate` and `grid_search` (multiprocessing-capable).

## Conventions and gotchas

- Outcomes are encoded as float: win=1.0, draw=0.5, loss=0.0. Probabilities are always P(competitor 1 wins).
- Ruff: single quotes, line length 120, google-style docstrings (rendered by pdoc).
- jax is an **optional extra** (`riix[jax]`), used only by `models/autograd_rating_system.py`; it is in the dev dependency group so tests cover it. Guard any new jax imports the same way (try/except with install hint).
- `tests/test_all_models.py` auto-discovers every `OnlineRatingSystem` subclass in `riix/models/` and smoke-tests fit+predict. A new model gets tested for free; if its constructor requires args beyond `competitors`, add them to `EXTRA_KWARGS` there.
- The README example is the de facto quickstart — if you rename public functions, update it and keep it runnable (it uses `generate_pair_data`).
- License is CC BY-NC-SA (intentionally non-commercial); don't suggest relicensing to permissive.
