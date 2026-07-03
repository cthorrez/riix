# Roadmap

Working list of improvements, roughly ordered within each horizon. (Started July 2026.)

## Short term

- [ ] **Finish the static/non-dynamic rating system inheritance refactor** (in flight — `RatingSystem` vs `OnlineRatingSystem` split in `core/base.py`).
- [ ] **Deduplicate model boilerplate into the base class** as part of that refactor:
  - `print_leaderboard` is copy-pasted in ~17 models, differing only in the sort key → generic base implementation over an overridable score array.
  - The `update_method` dispatch block (`'online'`/`'batched'` → bind `self.update`) is duplicated in ~20 `__init__`s → hoist into `OnlineRatingSystem.__init__`.
- [ ] **Normalize `fit_batch`/`fit_dataset` return values.** The None/array/tuple polymorphism is error-prone (already caused one shipped bug). Return a consistent shape or small result object.
- [ ] **`models/__init__.py` cleanup**: docstring lists a stale roster of systems; consider lazy re-exports (`from riix.models import Elo`) via module `__getattr__` to avoid importing every model (and optional deps) eagerly.
- [ ] **More known-value tests**: Elo and Weng-Lin against reference implementations, mirroring the existing Glicko/Glicko2/TrueSkill tests.

## Medium term

- [ ] **Two-team competitions** (promised in the README; >2 teams explicitly out of scope).
- [ ] **Benchmark suite** (`benchmarks/`): timing + accuracy comparisons across riix models and vs other packages (openskill.py, trueskill, etc.). Speed is the package's core pitch — measure and publish it.
  - [x] Internal speed benchmarks with branch A/B comparison (`benchmarks/bench.py`, `ab.py`, `compare.py` — July 2026)
  - [ ] Cross-package comparisons and accuracy benchmarks
- [ ] **Better hyperparameter search**: random search / Optuna (optional dep) alongside `grid_search`; progress reporting for long sweeps.
- [ ] **Type hints + `py.typed`** for the public API.
- [ ] **Test/lint CI workflow** (deliberately deferred for now) with a Python-version matrix matching `requires-python`.
- [ ] **Docs pass**: quickstart page, per-model docstrings with paper links, document the dataset format contract (sorted-by-time, outcome encoding).

## Long term

- [ ] **API stabilization → 0.1.0**: settle the base-class contract (predict/update/fit signatures, caching), then commit to semver.
- [ ] **New rating systems**: Whole History Rating (WHR), decayed-history Elo variants, other batch/offline systems enabled by the static-system refactor.
- [ ] **Release automation**: PyPI trusted publishing (OIDC), single gated release workflow (test → build once → TestPyPI → PyPI).
- [ ] **Published benchmark study** (blog post or short paper) using the benchmark suite; doubles as marketing for the package.
