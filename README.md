Open source implementations of online rating systems focusing on efficiency for offline experimentation

## When to use riix
This package is designed to accelerate experiments studying and comparing rating systems. In the scenario where you have paired comparison datasets with a known number of competitors and time range, riix exploits that information to achieve fast runtimes. It's **not** useful in the streaming case where new data with new competitors are coming in. It supports exactly 1v1 competitions — a competitor can be a single player or a whole team rated as one unit, but rating individual players within multi-player teams is out of scope.


> I have a large dataset of player matches for a game and want to determine which rating system out of Elo, Glicko, TrueSkill etc. gives the best predictive accuracy.

Use riix! 👍

> I want to incorporate skill based matchmaking into the game I am creating and want a package to compute ratings for players on the fly.

There are lots of other great python packages for that too! (just not riix)
* [openskill.py](https://github.com/OpenDebates/openskill.py) Multi-way competitions with [Weng-Lin](https://www.jmlr.org/papers/v12/weng11a.html) rating systems
* [trueskill](https://github.com/topics/trueskill) Open source implementation of Microsoft's rating system
* [PythonSkills](https://github.com/agoragames/PythonSkills) A port of [Moserware's](https://www.moserware.com/2010/03/computing-your-skill.html) Glicko and TrueSkill [C# code](https://github.com/moserware/Skills) to python
* [skelo](https://github.com/mbhynes/skelo/tree/main) Elo and Glicko 2 with the [scikit-learn](https://scikit-learn.org/stable/) interface
* [mmr-python](https://github.com/kari/mmr-python) Glicko and Weng-Lin rating systems
* glicko2
  * [sublee/glicko2](https://github.com/sublee/glicko2)
  * [deepy/glicko2](https://github.com/deepy/glicko2)
* [whole_history_rating](https://github.com/pfmonville/whole_history_rating) Python port of [WHR](https://www.remi-coulom.fr/WHR/)

## Installation

```bash
pip install riix
```

Requires Python ≥3.11. riix is built on [jax](https://docs.jax.dev), which is installed automatically; note that the jaxlib CPU wheel is large (~86 MB download, ~345 MB installed). GPU execution works by installing a CUDA-enabled jax variant (e.g. `pip install "jax[cuda12]"`) alongside riix.

## Example
```python
from riix.models.elo import Elo
from riix.utils import TimedPairDataset, split_pair_dataset, generate_pair_data
from riix.metrics import binary_metrics_suite

df = generate_pair_data() # replace with your **polars** dataframe
dataset = TimedPairDataset(
    df,
    competitor_cols=['competitor_1', 'competitor_2'],
    outcome_col='outcome',
    datetime_col='date',
    rating_period='1D',
)
train_dataset, test_dataset = split_pair_dataset(dataset, test_fraction=0.2)
print(f'{len(train_dataset)=}, {len(test_dataset)=}')

>>> Loaded dataset with:
>>> 10000 matchups
>>> 100 unique competitors
>>> 10 rating periods of length 1D
>>> Split into train_dataset of length 8000 and test_dataset of length 2000
>>> len(train_dataset)=8000, len(test_dataset)=2000

model = Elo(competitors=dataset.competitors)
model.fit_dataset(train_dataset)  # a second fit continues exactly like one fit of the concatenation
test_probs = model.fit_dataset(test_dataset, return_pre_match_probs=True).probs
test_metrics = binary_metrics_suite(probs=test_probs, outcomes=test_dataset.outcomes)
print(test_metrics)

>>> {'accuracy': 0.72825, 'accuracy_without_draws': 0.728592889334001, 'log_loss': 0.5383001523548737, 'brier_score': 0.1802256903116116}

model.print_leaderboard(num_places=5)

>>> competitor   	rating
>>> competitor_69	1874.170044
>>> competitor_75	1827.933472
>>> competitor_12	1826.119751
>>> competitor_81	1825.071777
>>> competitor_30	1802.338867
```

Returned probabilities are **prequential** by default — match *i* is predicted from the state after match *i-1*. Pass `predict_mode='period_start'` to instead predict every match in a rating period from period-start state: that is the honest protocol when your time granularity is a date and within-period order is arbitrary, since prequential probabilities leak within-period ordering. `update_method='batched'` applies one simultaneous update per rating period (on the systems that support it) and is inherently period-start.

## Design and performance

Every rating system runs on a jax `jit` + `lax.scan` backend (since 0.1.0):

- **One compiled scan per fit.** `fit_dataset` lowers the entire dataset into a single `lax.scan` whose step processes one match: gather the two competitors' states, predict, update, scatter back. No per-match or per-period python loop.
- **Compile once, fit many.** Hyperparameters are traced, so the compiled kernel is cached across model instances and hyperparameter values — one compile per (model class, mode, dtype, dataset shape). The first fit pays ~0.5-1s of compilation; every configuration of a hyperparameter sweep after that runs warm. `riix.eval.grid_search` / `random_search` / `optuna_search` exploit this; single-process sweeps are usually fastest since workers would each recompile.
- **Time dynamics precomputed outside jax.** Rating-deviation inflation (Glicko, Glicko2, TrueSkill, ...) comes from a vectorized numpy pre-pass, keeping the scan carry a small tuple of float arrays.
- **float32 by default.** For float64, pass `dtype=jnp.float64` to the constructor *and* run `jax.config.update('jax_enable_x64', True)` before fitting.

Warm fits run 2-30x faster than the numpy implementations they replaced, and batched-mode fits of a million matches land in tens of milliseconds; see [benchmarks/RESULTS.md](benchmarks/RESULTS.md) for measured comparisons against other Python rating system packages and [benchmarks/README.md](benchmarks/README.md) for the harness.

## License
This package is licensed under a
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa].
I've chosen a non-commercial license as overbroad protection to prevent the use of this package in the gambling and odds setting industries. If you would like to use riix for your business in any other area please do not hesitate to reach out and I'll happily grant you an eternal lifetime license. :)

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/

## About the name
The name riix represents an attempt to cleverly represent the idea of "R8" (pronounced "rate") alphabetically using the Roman numeral IIX in place of 8. By the time I realized the correct numeral would have been VIII I was already attatched to the name riix so I stuck with it. However on further research it turns out the Romans themselves occasionally used this form as well! [Why Romans Sometimes Wrote 8 as VIII, And Sometimes as IIX: A Possible Explanation](https://scholarworks.utep.edu/cs_techrep/1555/)

