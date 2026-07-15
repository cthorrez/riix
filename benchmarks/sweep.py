"""
Benchmark a realistic single-process hyperparameter sweep through riix.eval.grid_search.

Hyperparameters are *traced* jnp scalars, so the whole sweep pays one jit
compilation on its first fit and every later configuration runs at warm-scan
speed. To show that amortization, each sweep is timed twice in the same
process: "cold" includes the one-time compile, "warm" is a rerun of the
identical sweep hitting the compiled scan for every configuration.

--vmap additionally runs a prototype that `vmap`s the Elo scan over the whole
hyperparameter vector inside one jit (no public API; an experiment kept to
size the win in case sweeps ever become a bottleneck).

Usage:
    uv run python benchmarks/sweep.py
    uv run python benchmarks/sweep.py --num-matchups 100000 --num-configs 16 --models elo --vmap
"""

import argparse
import math
import time

import numpy as np
import jax
import jax.numpy as jnp

from riix.eval import grid_search
from riix.models.elo import Elo
from riix.models.elo_davidson import EloDavidson
from riix.models.glicko2 import Glicko2

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from bench import make_dataset  # noqa: E402


def elo_configs(n):
    return [{'k': float(k)} for k in np.geomspace(1.0, 256.0, n)]


def glicko2_configs(n):
    return [{'tau': float(t)} for t in np.geomspace(0.04, 1.2, n)]


MODELS = {
    'elo': (Elo, elo_configs),
    'elo_davidson': (EloDavidson, elo_configs),
    'glicko2': (Glicko2, glicko2_configs),
}


def time_sweep(rating_system_class, dataset, configurations):
    start = time.perf_counter()
    grid_search(rating_system_class, dataset, configurations)
    return time.perf_counter() - start


def vmap_elo_sweep(dataset, ks, num_competitors, dtype=jnp.float32):
    """Prototype: fit every k at once with one vmapped scan inside one jit."""
    matchups = jnp.asarray(dataset.matchups, dtype=jnp.int32)
    outcomes = jnp.asarray(dataset.outcomes, dtype=dtype)
    alpha = jnp.asarray(math.log(10.0) / 400.0, dtype=dtype)

    @jax.jit
    def sweep(ks, matchups, outcomes):
        def fit_one(k):
            state = (jnp.full(num_competitors, 1500.0, dtype=dtype),)
            hparams = {'k': k, 'alpha': alpha}

            def body(carry, x):
                return Elo._step(carry, x, hparams)

            _, probs = jax.lax.scan(body, state, (matchups, outcomes))
            return probs

        return jax.vmap(fit_one)(ks)

    return sweep(jnp.asarray(ks, dtype=dtype), matchups, outcomes)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--num-matchups', type=int, default=100_000)
    parser.add_argument('--num-competitors', type=int, default=1_000)
    parser.add_argument('--rating-periods', type=int, default=10)
    parser.add_argument('--num-configs', type=int, default=16)
    parser.add_argument('--models', type=str, default=','.join(MODELS))
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--vmap', action='store_true', help='also run the vmapped Elo sweep prototype')
    args = parser.parse_args()

    dataset = make_dataset(args.num_matchups, args.num_competitors, args.rating_periods, args.seed)
    n = args.num_configs
    print(f'{args.num_matchups} matchups, {args.num_competitors} competitors, {n} configurations, single process')
    header = (
        f'{"model": <14} {"cold total (s)": >15} {"warm total (s)": >15} '
        f'{"cold s/config": >14} {"warm s/config": >14}'
    )
    print(header)
    print('-' * len(header))

    for name in args.models.split(','):
        model_class, make_configs = MODELS[name]
        configurations = make_configs(n)
        cold = time_sweep(model_class, dataset, configurations)
        warm = time_sweep(model_class, dataset, configurations)
        print(f'{name: <14} {cold: >15.2f} {warm: >15.2f} {cold / n: >14.3f} {warm / n: >14.3f}')

    if args.vmap:
        ks = [c['k'] for c in elo_configs(n)]
        start = time.perf_counter()
        probs = vmap_elo_sweep(dataset, ks, dataset.num_competitors)
        probs.block_until_ready()
        cold = time.perf_counter() - start
        start = time.perf_counter()
        probs = vmap_elo_sweep(dataset, ks, dataset.num_competitors)
        probs.block_until_ready()
        warm = time.perf_counter() - start
        print(f'vmap elo prototype ({n} configs in one jit): cold {cold:.2f}s, warm {warm:.2f}s')


if __name__ == '__main__':
    main()
