"""
Compare two benchmark result files produced by bench.py.

Usage:
    uv run python benchmarks/compare.py base.json head.json
"""

import argparse
import json
import math


def load(path):
    with open(path) as f:
        data = json.load(f)
    keyed = {(r['model'], r['num_matchups'], r['num_competitors']): r for r in data['results']}
    return data['meta'], keyed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('base', help='JSON results for the baseline (e.g. main)')
    parser.add_argument('head', help='JSON results for the candidate change')
    parser.add_argument('--threshold', type=float, default=0.05, help='relative change considered significant')
    args = parser.parse_args()

    base_meta, base = load(args.base)
    head_meta, head = load(args.head)
    print(f'base: {base_meta["git"]["branch"]} @ {str(base_meta["git"]["sha"])[:8]} ({args.base})')
    print(f'head: {head_meta["git"]["branch"]} @ {str(head_meta["git"]["sha"])[:8]} ({args.head})\n')

    common = sorted(base.keys() & head.keys())
    missing = base.keys() ^ head.keys()
    if missing:
        print(f'warning: {len(missing)} configs present in only one file are skipped\n')

    header = f'{"model": <32} {"matchups": >9} {"competitors": >12} {"base (s)": >10} {"head (s)": >10} {"speedup": >8}'
    print(header)
    print('-' * len(header))
    log_speedups = []
    for key in common:
        b, h = base[key]['median_seconds'], head[key]['median_seconds']
        speedup = b / h if h > 0 else math.inf
        log_speedups.append(math.log(speedup))
        marker = ''
        if speedup >= 1.0 + args.threshold:
            marker = ' faster'
        elif speedup <= 1.0 - args.threshold:
            marker = ' SLOWER'
        model, num_matchups, num_competitors = key
        print(
            f'{model: <32} {num_matchups: >9} {num_competitors: >12} {b: >10.4f} {h: >10.4f} {speedup: >7.2f}x{marker}'
        )

    if log_speedups:
        geo_mean = math.exp(sum(log_speedups) / len(log_speedups))
        print(f'\ngeometric mean speedup over {len(log_speedups)} configs: {geo_mean:.3f}x')


if __name__ == '__main__':
    main()
