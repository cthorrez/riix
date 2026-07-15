# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "riix",
#     "trueskill>=0.4.5",
#     "openskill>=6.0",
#     "glicko2>=2.0",
#     "elote>=1.1",
#     # chess and pyzstd are imported by elote's top-level __init__ but not declared as dependencies
#     "chess",
#     "pyzstd",
# ]
#
# [tool.uv.sources]
# riix = { path = "..", editable = true }
# ///
"""
Benchmark riix against other Python rating system packages on the same data.

Each task pairs a riix model with external implementations of the same
algorithm. All implementations process the identical synthetic match history:
riix via fit_dataset over rating periods, the external packages via their
idiomatic one-match-at-a-time update loop. riix appears twice per task —
update_method='online' (per-match sequential updates, the same semantics as
the external loops, and the speedup baseline) and 'batched' (one vectorized
update per rating period). Timing covers rating initialization plus processing
every matchup; dataset generation is untimed, and the default --warmup keeps
riix's one-time jit compilation out of the timed runs.

This measures fitting speed only, with each library's default hyperparameters.
It says nothing about predictive accuracy.

Usage (uv installs the external packages into an ephemeral script environment):
    uv run benchmarks/external.py
    uv run benchmarks/external.py --num-matchups 10000,100000 --tasks elo,trueskill -o results.json
"""

import argparse
import importlib.metadata
import json
import platform
import statistics
import time
import warnings
from dataclasses import dataclass

with warnings.catch_warnings():
    # trueskill 0.4.5 triggers SyntaxWarnings on modern Python; elote warns about optional deps
    warnings.simplefilter('ignore')
    import trueskill
    from elote import EloCompetitor, Glicko2Competitor, GlickoCompetitor
    from glicko2 import Player as Glicko2Player
    from openskill.models import BradleyTerryFull, ThurstoneMostellerFull

from bench import block_on_state, git_meta, int_list, make_dataset

from riix.models.elo import Elo
from riix.models.glicko import Glicko
from riix.models.glicko2 import Glicko2
from riix.models.trueskill import TrueSkill
from riix.models.weng_lin import WengLin

EXTERNAL_PACKAGES = ['trueskill', 'openskill', 'glicko2', 'elote']


@dataclass
class BenchData:
    """One synthetic match history in the two layouts the implementations consume."""

    dataset: object  # TimedPairDataset for riix
    matchups: list  # list of (idx_1, idx_2) tuples for per-match loops
    outcomes: list  # list of floats: win=1.0, draw=0.5, loss=0.0
    num_competitors: int


def make_bench_data(num_matchups, num_competitors, num_rating_periods, seed):
    dataset = make_dataset(num_matchups, num_competitors, num_rating_periods, seed)
    return BenchData(
        dataset=dataset,
        matchups=[tuple(m) for m in dataset.matchups.tolist()],
        outcomes=dataset.outcomes.tolist(),
        num_competitors=len(dataset.competitors),
    )


def riix_fit(model_class, **kwargs):
    def fit(data):
        model = model_class(competitors=data.dataset.competitors, **kwargs)
        model.fit_dataset(data.dataset)
        block_on_state(getattr(model, 'state', None))  # jax dispatch is async; numpy models no-op

    return fit


def trueskill_fit(data):
    env = trueskill.TrueSkill()
    ratings = [env.create_rating() for _ in range(data.num_competitors)]
    rate = env.rate_1vs1
    for (comp_1, comp_2), outcome in zip(data.matchups, data.outcomes):
        if outcome == 1.0:
            ratings[comp_1], ratings[comp_2] = rate(ratings[comp_1], ratings[comp_2])
        elif outcome == 0.0:
            ratings[comp_2], ratings[comp_1] = rate(ratings[comp_2], ratings[comp_1])
        else:
            ratings[comp_1], ratings[comp_2] = rate(ratings[comp_1], ratings[comp_2], drawn=True)


def openskill_fit(model_class):
    def fit(data):
        model = model_class()
        ratings = [model.rating() for _ in range(data.num_competitors)]
        rate = model.rate
        for (comp_1, comp_2), outcome in zip(data.matchups, data.outcomes):
            if outcome == 1.0:
                [[ratings[comp_1]], [ratings[comp_2]]] = rate([[ratings[comp_1]], [ratings[comp_2]]])
            elif outcome == 0.0:
                [[ratings[comp_2]], [ratings[comp_1]]] = rate([[ratings[comp_2]], [ratings[comp_1]]])
            else:
                [[ratings[comp_1]], [ratings[comp_2]]] = rate([[ratings[comp_1]], [ratings[comp_2]]], ranks=[1, 1])

    return fit


def glicko2_fit(data):
    players = [Glicko2Player() for _ in range(data.num_competitors)]
    for (comp_1, comp_2), outcome in zip(data.matchups, data.outcomes):
        p_1, p_2 = players[comp_1], players[comp_2]
        rating_1, rd_1 = p_1.rating, p_1.rd
        rating_2, rd_2 = p_2.rating, p_2.rd
        p_1.update_player([rating_2], [rd_2], [outcome])
        p_2.update_player([rating_1], [rd_1], [1.0 - outcome])


def elote_fit(competitor_class, **kwargs):
    def fit(data):
        competitors = [competitor_class(**kwargs) for _ in range(data.num_competitors)]
        for (comp_1, comp_2), outcome in zip(data.matchups, data.outcomes):
            if outcome == 1.0:
                competitors[comp_1].beat(competitors[comp_2])
            elif outcome == 0.0:
                competitors[comp_2].beat(competitors[comp_1])
            else:
                competitors[comp_1].tied(competitors[comp_2])

    return fit


# task -> list of (implementation label, fit function); the first entry is the speedup baseline.
# Each riix model appears twice: 'online' processes matchups sequentially exactly like the external
# per-match loops, 'batched' applies one vectorized update per rating period (with slightly
# different update semantics within a period).
TASKS = {
    'elo': [
        ('riix.Elo(online)', riix_fit(Elo, update_method='online')),
        ('riix.Elo(batched)', riix_fit(Elo, update_method='batched')),
        # start at 1500 like riix; elote's 400 default hits its minimum-rating floor on long histories
        ('elote.EloCompetitor', elote_fit(EloCompetitor, initial_rating=1500)),
    ],
    'glicko': [
        ('riix.Glicko(online)', riix_fit(Glicko, update_method='online')),
        ('riix.Glicko(batched)', riix_fit(Glicko, update_method='batched')),
        ('elote.GlickoCompetitor', elote_fit(GlickoCompetitor)),
    ],
    'glicko2': [
        ('riix.Glicko2(online)', riix_fit(Glicko2, update_method='online')),
        ('riix.Glicko2(batched)', riix_fit(Glicko2, update_method='batched')),
        ('glicko2.Player', glicko2_fit),
        ('elote.Glicko2Competitor', elote_fit(Glicko2Competitor)),
    ],
    'trueskill': [
        ('riix.TrueSkill(online)', riix_fit(TrueSkill, update_method='online')),
        ('riix.TrueSkill(batched)', riix_fit(TrueSkill, update_method='batched')),
        ('trueskill.TrueSkill', trueskill_fit),
    ],
    'wenglin_bt': [
        ('riix.WengLin(bt, online)', riix_fit(WengLin, model='bt', update_method='online')),
        ('riix.WengLin(bt, batched)', riix_fit(WengLin, model='bt', update_method='batched')),
        ('openskill.BradleyTerryFull', openskill_fit(BradleyTerryFull)),
    ],
    'wenglin_tm': [
        ('riix.WengLin(tm, online)', riix_fit(WengLin, model='tm', update_method='online')),
        ('riix.WengLin(tm, batched)', riix_fit(WengLin, model='tm', update_method='batched')),
        ('openskill.ThurstoneMostellerFull', openskill_fit(ThurstoneMostellerFull)),
    ],
}


def time_fit(fit, data, repeats, warmup):
    times = []
    for i in range(warmup + repeats):
        start = time.perf_counter()
        fit(data)
        elapsed = time.perf_counter() - start
        if i >= warmup:
            times.append(elapsed)
    return times


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--num-matchups', type=int_list, default=[10_000])
    parser.add_argument('--num-competitors', type=int_list, default=[100])
    parser.add_argument('--rating-periods', type=int, default=10)
    parser.add_argument('--repeats', type=int, default=3, help='timed runs per config, median is reported')
    parser.add_argument('--warmup', type=int, default=1, help='untimed runs per config')
    parser.add_argument('--tasks', type=str, default=None, help='comma separated substrings to filter task names')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('-o', '--output', type=str, default=None, help='write results to this JSON file')
    args = parser.parse_args()

    tasks = TASKS
    if args.tasks:
        patterns = [p.lower() for p in args.tasks.split(',')]
        tasks = {n: impls for n, impls in tasks.items() if any(p in n.lower() for p in patterns)}
    if not tasks:
        raise SystemExit(f'no tasks matched, available: {", ".join(TASKS)}')

    results = []
    for num_matchups in args.num_matchups:
        for num_competitors in args.num_competitors:
            data = make_bench_data(num_matchups, num_competitors, args.rating_periods, args.seed)
            for task, impls in tasks.items():
                print(f'\ntask={task} matchups={num_matchups} competitors={num_competitors}')
                baseline_median = None
                for idx, (label, fit) in enumerate(impls):
                    times = time_fit(fit, data, args.repeats, args.warmup)
                    median = statistics.median(times)
                    if idx == 0:  # first impl per task is riix online
                        baseline_median = median
                    results.append(
                        {
                            'task': task,
                            'impl': label,
                            'num_matchups': num_matchups,
                            'num_competitors': num_competitors,
                            'median_seconds': median,
                            'min_seconds': min(times),
                            'times': times,
                        }
                    )
                    relative = ' ' * 8 + 'baseline' if idx == 0 else f'{median / baseline_median:>8.2f}x baseline'
                    print(f'  {label: <36} {median: >9.4f}s {relative}')

    output = {
        'meta': {
            'git': git_meta(),
            'python': platform.python_version(),
            'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
            'packages': {pkg: importlib.metadata.version(pkg) for pkg in EXTERNAL_PACKAGES},
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
