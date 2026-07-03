# Roadmap

Working list of improvements, roughly ordered within each horizon. (Started July 2026.)

## Short term

- [x] **Finish the static/non-dynamic rating system inheritance refactor** (`RatingSystem` → `OnlineRatingSystem`/`StaticRatingSystem` split in `core/base.py`; static systems fit once on the whole dataset and share the `fit_dataset` contract so `riix.eval` works with both — July 2026)
- [x] **Deduplicate model boilerplate into the base class** as part of that refactor:
  - `print_leaderboard` is now a generic base implementation over an overridable `leaderboard_ratings()` array + `leaderboard_header` (July 2026)
  - The `update_method` dispatch block is hoisted into `OnlineRatingSystem.__init__` (July 2026)
- [x] **Normalize `fit_batch`/`fit_dataset` return values.** Both always return a `FitResult` NamedTuple with `probs`/`ratings` fields that are None unless requested (July 2026)
- [x] **`models/__init__.py` cleanup**: lazy re-exports via module `__getattr__`, accurate roster in the docstring (July 2026)
- [x] **More known-value tests**: Elo vs hand-computed classic arithmetic, Weng-Lin BT vs openskill.py 6.2.0; plus batched==online regression tests for the fixed OnlineDiscDecomp/EloMentum batched updates (July 2026)

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
