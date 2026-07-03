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

## How does riix compare to other rating system packages?

See [RESULTS.md](RESULTS.md) for numbers from a recent run.

```bash
uv run benchmarks/external.py
uv run benchmarks/external.py --num-matchups 10000,100000 --tasks elo,trueskill -o results.json
```

`external.py` benchmarks riix against other Python implementations of the same algorithms on identical synthetic data:

| task | riix model | external implementations |
| --- | --- | --- |
| elo | `Elo` | [elote](https://github.com/wdm0006/elote) `EloCompetitor` |
| glicko | `Glicko` | elote `GlickoCompetitor` |
| glicko2 | `Glicko2` | [glicko2](https://github.com/deepy/glicko2) `Player`, elote `Glicko2Competitor` |
| trueskill | `TrueSkill` | [trueskill](https://github.com/sublee/trueskill) `rate_1vs1` |
| wenglin_bt / wenglin_tm | `WengLin` | [openskill](https://github.com/vivekjoshy/openskill.py) `BradleyTerryFull` / `ThurstoneMostellerFull` |

The script has [inline dependency metadata](https://docs.astral.sh/uv/guides/scripts/#declaring-script-dependencies), so `uv run` (without `python`) installs the external packages into an ephemeral environment instead of the project one, with riix itself taken from the working tree.

Each riix model is run twice: `update_method='online'` processes matchups sequentially — the same semantics as the external packages' one-match-at-a-time loops, and the printed speedup baseline — while `'batched'` applies one vectorized update per rating period, riix's headline mode. This measures fitting speed only, with each library's default hyperparameters; it says nothing about predictive accuracy. Candidates considered but skipped: [skelo](https://github.com/mbhynes/skelo) (sklearn-interface Elo/Glicko2; fit loop is dominated by pandas overhead and it has no draw support) and [PythonSkills](https://github.com/McLeopold/PythonSkills) (unmaintained since 2012).
