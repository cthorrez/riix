# Roadmap

Working list of improvements, roughly ordered within each horizon. (Started July 2026; rewritten after the jax migration landed, July 2026.)

## Done (July 2026)

- [x] **Base-class refactor**: `RatingSystem` → `OnlineRatingSystem`/`StaticRatingSystem` split, generic `print_leaderboard`, normalized `FitResult` returns
- [x] **Benchmark suite** (`benchmarks/`): internal timing with branch A/B comparison (`bench.py`, `ab.py`), cross-package speed (`external.py`) and accuracy (`accuracy.py`) comparisons, published in `benchmarks/RESULTS.md`
- [x] **Better hyperparameter search**: `random_search` and `optuna_search` (optional extra) alongside `grid_search`, shared distribution specs, `verbose=` progress
- [x] **The jax migration (0.1.0)**: every rating system rewritten on a jit + `lax.scan` backend — one compiled scan per fit, traced hyperparameters so sweeps compile once, `predict_mode='prequential'`/`'period_start'`, padded-period `update_method='batched'`, exact train → test continuation — plus the v1 quirk fixes (corrected TrueSkill/WengLin draw moments, first-appearance inflation, capped batched phi_star, `get_pre_match_ratings` everywhere meaningful, period-aligned dataset splitting). History and decisions in `MIGRATION.md`, user-facing summary in `CHANGELOG.md`
- [x] ~~**Two-team competitions**~~ — descoped: riix stays strictly 1v1. A "competitor" can already be a whole team rated as a single unit; per-player ratings within teams are out of scope

## Short term

- [ ] **Release 0.1.0** to PyPI (changelog is finalized; verify the pdoc CI build renders the model docs on merge)
- [ ] **Test/lint CI workflow** with a Python-version matrix matching `requires-python` (≥3.11)
- [ ] **Type hints + `py.typed`** for the public API

## Medium term

- [ ] **`vmap` sweep API**: vmap the scan over a hyperparameter axis inside one jit — the prototype in `benchmarks/sweep.py --vmap` is ~4x faster than a sequential warm sweep on CPU (~10ms per Elo configuration). Promote to a public API if sweeps become a bottleneck; likely bigger wins on GPU
- [ ] **GPU benchmarks**: the backend runs on GPU by installing `jax[cuda12]`, but nothing is measured yet — benchmark large-dataset fits and vmap sweeps
- [ ] **Docs pass**: document the dataset format contract (chronologically sorted, win=1.0/draw=0.5/loss=0.0, fixed competitor set) and add paper links to the remaining model docstrings
- [ ] **New rating systems**: Whole History Rating (WHR), decayed-history Elo variants, other batch/offline systems enabled by the static-system split

## Long term

- [ ] **Semver commitment** once 0.1.x has settled the post-migration API surface
- [ ] **Release automation**: PyPI trusted publishing (OIDC), single gated release workflow (test → build once → TestPyPI → PyPI)
- [ ] **Published benchmark study** (blog post or short paper) using the benchmark suite; the EsportsBench sweep replication (2-12x per rating system at realistic sweep sizes) is a ready-made case study
