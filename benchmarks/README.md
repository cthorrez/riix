# Speed benchmarks

Internal tools for measuring `fit_dataset` wall time of every rating system on synthetic data. Not shipped with the package.

## Benchmark the current checkout

```bash
uv run python benchmarks/bench.py
uv run python benchmarks/bench.py --num-matchups 1000,10000,100000 --num-competitors 100,1000 --models Elo,Glicko -o results.json
```

Models are auto-discovered from `riix/models/`. Timing covers `fit_dataset` only (not model construction or dataset generation); the median of `--repeats` runs is reported after `--warmup` untimed runs. Data is seeded, so runs are comparable across branches.

## Did my change make it faster?

```bash
# compare your working tree (including uncommitted changes) against main
uv run python benchmarks/ab.py

# or compare two branches / any two refs
uv run python benchmarks/ab.py --base main --head my-feature-branch

# pass through bench options to keep it quick while iterating
uv run python benchmarks/ab.py --models Elo --num-matchups 100000 --repeats 5
```

`ab.py` checks the base ref out into a temporary git worktree with its own uv environment, runs the same benchmark on both sides, and prints per-config speedups plus a geometric mean. The current `bench.py` is used for both sides, so the measurement logic is identical even if the base ref predates it.

You can also run the two halves manually with `bench.py -o base.json` / `-o head.json` and `uv run python benchmarks/compare.py base.json head.json`.

Tips for stable numbers: close heavy background processes, prefer `--repeats 5` for small configs, and be skeptical of changes under ~5% (the default significance threshold in `compare.py`).
