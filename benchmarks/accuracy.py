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
Benchmark the predictive accuracy of every riix rating system, and of external
packages implementing the same algorithms, on identical synthetic data.

Every implementation is evaluated prequentially: each match is predicted from
the ratings fit on all earlier matches, then used to update them. To make riix
and the external per-match loops semantically identical, the data uses one
rating period per match, so riix's per-period time dynamics fire per match just
like the external packages' per-update dynamics. Metrics come from
riix.metrics.binary_metrics_suite over all matches; every implementation uses
its own default hyperparameters, so differences reflect both the algorithm and
the defaults.

riix models are auto-discovered from riix/models/ (like bench.py) and run at
their defaults (float32, prequential — which is also exactly how the external
per-match loops evaluate). External packages cover the same algorithms as
external.py: trueskill, openskill (Bradley-Terry and Thurstone-Mosteller
full), glicko2, and elote (Elo, Glicko, Glicko2).

Usage (uv installs the external packages into an ephemeral script environment):
    uv run benchmarks/accuracy.py
    uv run benchmarks/accuracy.py --num-matchups 100000 --impls riix,openskill -o results.json
"""

import argparse
import importlib.metadata
import json
import math
import platform
import statistics
import time
import warnings

import numpy as np

with warnings.catch_warnings():
    # trueskill 0.4.5 triggers SyntaxWarnings on modern Python; elote warns about optional deps
    warnings.simplefilter('ignore')
    import trueskill
    from elote import EloCompetitor, Glicko2Competitor, GlickoCompetitor
    from glicko2 import Player as Glicko2Player
    from openskill.models import BradleyTerryFull, ThurstoneMostellerFull
    from scipy.stats import norm

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from bench import EXTRA_KWARGS, discover_model_classes, git_meta, int_list  # noqa: E402

from riix.metrics import binary_metrics_suite
from riix.utils import generate_pair_data
from riix.utils.data_utils import TimedPairDataset

EXTERNAL_PACKAGES = ['trueskill', 'openskill', 'glicko2', 'elote']
GLICKO2_SCALE = 173.7178
THREE_OVER_PI_SQUARED = 3.0 / (math.pi**2.0)


def make_data(num_matchups, num_competitors, seed):
    """One rating period per match, so riix online fits are exactly prequential."""
    df = generate_pair_data(num_matchups=num_matchups, num_competitors=num_competitors, seed=seed)
    grouped = TimedPairDataset(
        df,
        competitor_cols=['competitor_1', 'competitor_2'],
        outcome_col='outcome',
        datetime_col='date',
        rating_period='1D',
        verbose=False,
    )
    return TimedPairDataset.init_from_arrays(
        time_steps=np.arange(len(grouped)),
        matchups=grouped.matchups,
        outcomes=grouped.outcomes,
        competitors=grouped.competitors,
    )


def riix_eval(model_class, **kwargs):
    def run(dataset):
        model = model_class(competitors=dataset.competitors, **kwargs)
        return model.fit_dataset(dataset, return_pre_match_probs=True).probs

    return run


def trueskill_eval(dataset):
    env = trueskill.TrueSkill()
    ratings = [env.create_rating() for _ in range(dataset.num_competitors)]
    two_beta2 = 2.0 * env.beta**2.0
    probs = np.empty(len(dataset))
    for i, ((comp_1, comp_2), outcome) in enumerate(zip(dataset.matchups.tolist(), dataset.outcomes.tolist())):
        r_1, r_2 = ratings[comp_1], ratings[comp_2]
        probs[i] = norm.cdf((r_1.mu - r_2.mu) / math.sqrt(two_beta2 + r_1.sigma**2.0 + r_2.sigma**2.0))
        if outcome == 1.0:
            ratings[comp_1], ratings[comp_2] = env.rate_1vs1(r_1, r_2)
        elif outcome == 0.0:
            ratings[comp_2], ratings[comp_1] = env.rate_1vs1(r_2, r_1)
        else:
            ratings[comp_1], ratings[comp_2] = env.rate_1vs1(r_1, r_2, drawn=True)
    return probs


def openskill_eval(model_class):
    def run(dataset):
        model = model_class()
        ratings = [model.rating() for _ in range(dataset.num_competitors)]
        rate = model.rate
        predict_win = model.predict_win
        probs = np.empty(len(dataset))
        for i, ((comp_1, comp_2), outcome) in enumerate(zip(dataset.matchups.tolist(), dataset.outcomes.tolist())):
            teams = [[ratings[comp_1]], [ratings[comp_2]]]
            probs[i] = predict_win(teams)[0]
            if outcome == 1.0:
                [[ratings[comp_1]], [ratings[comp_2]]] = rate(teams)
            elif outcome == 0.0:
                [[ratings[comp_2]], [ratings[comp_1]]] = rate([teams[1], teams[0]])
            else:
                [[ratings[comp_1]], [ratings[comp_2]]] = rate(teams, ranks=[1, 1])
        return probs

    return run


def glicko2_prob(player_1, player_2):
    """Glicko-2 expected score from the display-scale rating/rd, like riix.Glicko2.predict."""
    mu_diff = (player_1.rating - player_2.rating) / GLICKO2_SCALE
    phi2_sum = (player_1.rd**2.0 + player_2.rd**2.0) / GLICKO2_SCALE**2.0
    g = 1.0 / math.sqrt(1.0 + phi2_sum * THREE_OVER_PI_SQUARED)
    return 1.0 / (1.0 + math.exp(-g * mu_diff))


def glicko2_eval(dataset):
    players = [Glicko2Player() for _ in range(dataset.num_competitors)]
    probs = np.empty(len(dataset))
    for i, ((comp_1, comp_2), outcome) in enumerate(zip(dataset.matchups.tolist(), dataset.outcomes.tolist())):
        p_1, p_2 = players[comp_1], players[comp_2]
        probs[i] = glicko2_prob(p_1, p_2)
        rating_1, rd_1 = p_1.rating, p_1.rd
        rating_2, rd_2 = p_2.rating, p_2.rd
        p_1.update_player([rating_2], [rd_2], [outcome])
        p_2.update_player([rating_1], [rd_1], [1.0 - outcome])
    return probs


def elote_eval(competitor_class, **kwargs):
    def run(dataset):
        competitors = [competitor_class(**kwargs) for _ in range(dataset.num_competitors)]
        probs = np.empty(len(dataset))
        for i, ((comp_1, comp_2), outcome) in enumerate(zip(dataset.matchups.tolist(), dataset.outcomes.tolist())):
            probs[i] = competitors[comp_1].expected_score(competitors[comp_2])
            if outcome == 1.0:
                competitors[comp_1].beat(competitors[comp_2])
            elif outcome == 0.0:
                competitors[comp_2].beat(competitors[comp_1])
            else:
                competitors[comp_1].tied(competitors[comp_2])
        return probs

    return run


def build_impls():
    impls = {}
    for name, model_class in sorted(discover_model_classes().items()):
        impls[f'riix.{name}'] = riix_eval(model_class, **EXTRA_KWARGS.get(model_class.__name__, {}))
    impls['trueskill.TrueSkill'] = trueskill_eval
    impls['openskill.BradleyTerryFull'] = openskill_eval(BradleyTerryFull)
    impls['openskill.ThurstoneMostellerFull'] = openskill_eval(ThurstoneMostellerFull)
    impls['glicko2.Player'] = glicko2_eval
    # start at 1500 like riix; elote's 400 default hits its minimum-rating floor on long histories
    impls['elote.EloCompetitor'] = elote_eval(EloCompetitor, initial_rating=1500)
    impls['elote.GlickoCompetitor'] = elote_eval(GlickoCompetitor)
    impls['elote.Glicko2Competitor'] = elote_eval(Glicko2Competitor)
    return impls


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--num-matchups', type=int_list, default=[10_000])
    parser.add_argument('--num-competitors', type=int_list, default=[100])
    parser.add_argument('--repeats', type=int, default=1, help='timed runs per impl, median seconds is reported')
    parser.add_argument('--warmup', type=int, default=1, help='untimed runs per impl (keeps one-time jit compilation out of the timings)')
    parser.add_argument('--impls', type=str, default=None, help='comma separated substrings to filter impl names')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('-o', '--output', type=str, default=None, help='write results to this JSON file')
    args = parser.parse_args()

    impls = build_impls()
    if args.impls:
        patterns = [p.lower() for p in args.impls.split(',')]
        impls = {n: fn for n, fn in impls.items() if any(p in n.lower() for p in patterns)}
    if not impls:
        raise SystemExit(f'no impls matched, available: {", ".join(build_impls())}')

    results = []
    for num_matchups in args.num_matchups:
        for num_competitors in args.num_competitors:
            dataset = make_data(num_matchups, num_competitors, args.seed)
            rows = []
            for name, fn in impls.items():
                times, probs = [], None
                for i in range(args.warmup + args.repeats):
                    start = time.perf_counter()
                    probs = fn(dataset)
                    if i >= args.warmup:
                        times.append(time.perf_counter() - start)
                metrics = binary_metrics_suite(np.asarray(probs), dataset.outcomes)
                rows.append(
                    {
                        'impl': name,
                        'num_matchups': num_matchups,
                        'num_competitors': num_competitors,
                        'median_seconds': statistics.median(times),
                        **metrics,
                    }
                )
            rows.sort(key=lambda r: r['log_loss'])
            results.extend(rows)

            print(f'\nmatchups={num_matchups} competitors={num_competitors} (sorted by log loss)')
            header = f'{"impl": <36} {"log_loss": >9} {"accuracy": >9} {"brier": >9} {"seconds": >9}'
            print(header)
            print('-' * len(header))
            for row in rows:
                print(
                    f'{row["impl"]: <36} {row["log_loss"]: >9.4f} {row["accuracy"]: >9.4f} '
                    f'{row["brier_score"]: >9.4f} {row["median_seconds"]: >9.4f}'
                )

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
