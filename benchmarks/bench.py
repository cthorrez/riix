"""
Benchmark the fit speed of riix rating systems on synthetic data.

Runs every discovered model over a grid of (num_matchups, num_competitors)
configs and reports fit_dataset wall time. Results can be written to JSON
for comparison across branches with compare.py / ab.py.

Usage:
    uv run python benchmarks/bench.py
    uv run python benchmarks/bench.py --num-matchups 1000,10000,100000 --num-competitors 100,1000 -o results.json
    uv run python benchmarks/bench.py --models Elo,Glicko --repeats 5
"""

import argparse
import importlib
import inspect
import json
import pkgutil
import platform
import statistics
import subprocess
import time

import riix.models
from riix.core.base import OnlineRatingSystem
from riix.utils import generate_pair_data
from riix.utils.data_utils import TimedPairDataset

# models whose constructors require more than `competitors`
EXTRA_KWARGS = {
    'BaselineRatingSystem': {'mode': 'win_rate'},
}
EXCLUDED = {'Template'}


def discover_model_classes():
    classes = {}
    for mod_info in pkgutil.iter_modules(riix.models.__path__):
        try:
            module = importlib.import_module(f'riix.models.{mod_info.name}')
        except ImportError:
            continue
        for name, obj in inspect.getmembers(module, inspect.isclass):
            if issubclass(obj, OnlineRatingSystem) and obj.__module__ == module.__name__ and name not in EXCLUDED:
                classes[name] = obj
    return classes


def make_dataset(num_matchups, num_competitors, num_rating_periods, seed):
    df = generate_pair_data(
        num_matchups=num_matchups,
        num_competitors=num_competitors,
        num_rating_periods=num_rating_periods,
        seed=seed,
    )
    return TimedPairDataset(
        df,
        competitor_cols=['competitor_1', 'competitor_2'],
        outcome_col='outcome',
        datetime_col='date',
        rating_period='1D',
        verbose=False,
    )


def time_fit(model_class, kwargs, dataset, repeats, warmup):
    """Time fit_dataset on fresh model instances. Model construction is not timed."""
    times = []
    for i in range(warmup + repeats):
        model = model_class(competitors=dataset.competitors, **kwargs)
        start = time.perf_counter()
        model.fit_dataset(dataset)
        elapsed = time.perf_counter() - start
        if i >= warmup:
            times.append(elapsed)
    return times


def git_meta():
    def run(args):
        try:
            return subprocess.run(args, capture_output=True, text=True, check=True).stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return None

    return {
        'sha': run(['git', 'rev-parse', 'HEAD']),
        'branch': run(['git', 'rev-parse', '--abbrev-ref', 'HEAD']),
        'dirty': bool(run(['git', 'status', '--porcelain'])),
    }


def int_list(arg):
    return [int(x) for x in arg.split(',')]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--num-matchups', type=int_list, default=[1_000, 10_000, 100_000])
    parser.add_argument('--num-competitors', type=int_list, default=[100, 1_000])
    parser.add_argument('--rating-periods', type=int, default=10)
    parser.add_argument('--repeats', type=int, default=3, help='timed runs per config, median is reported')
    parser.add_argument('--warmup', type=int, default=1, help='untimed runs per config (JIT compilation, caches)')
    parser.add_argument('--models', type=str, default=None, help='comma separated substrings to filter model names')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('-o', '--output', type=str, default=None, help='write results to this JSON file')
    args = parser.parse_args()

    model_classes = discover_model_classes()
    if args.models:
        patterns = [p.lower() for p in args.models.split(',')]
        model_classes = {n: c for n, c in model_classes.items() if any(p in n.lower() for p in patterns)}
    if not model_classes:
        raise SystemExit('no models matched')

    results = []
    for num_matchups in args.num_matchups:
        for num_competitors in args.num_competitors:
            dataset = make_dataset(num_matchups, num_competitors, args.rating_periods, args.seed)
            for name, model_class in sorted(model_classes.items()):
                kwargs = EXTRA_KWARGS.get(name, {})
                times = time_fit(model_class, kwargs, dataset, args.repeats, args.warmup)
                median = statistics.median(times)
                results.append(
                    {
                        'model': name,
                        'num_matchups': num_matchups,
                        'num_competitors': num_competitors,
                        'median_seconds': median,
                        'min_seconds': min(times),
                        'times': times,
                    }
                )
                print(f'{name: <32} matchups={num_matchups: <8} competitors={num_competitors: <6} {median:.4f}s')

    output = {
        'meta': {
            'git': git_meta(),
            'python': platform.python_version(),
            'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
            'args': {k: v for k, v in vars(args).items() if k != 'output'},
        },
        'results': results,
    }
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(output, f, indent=2)
        print(f'\nwrote {len(results)} results to {args.output}')


if __name__ == '__main__':
    main()
