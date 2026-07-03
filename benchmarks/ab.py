"""
A/B speed comparison between two git refs.

Benchmarks the base ref in a temporary git worktree (with its own uv
environment) and the head ref (default: your current working tree, including
uncommitted changes), then prints a per-config speedup table.

The *current* bench.py script is used for both sides so the measurement logic
is identical even if the base ref predates it.

Usage:
    uv run python benchmarks/ab.py                       # current tree vs main
    uv run python benchmarks/ab.py --base main --head my-branch
    uv run python benchmarks/ab.py --models Elo,Glicko --num-matchups 10000
"""

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

BENCH_ARGS = ['--num-matchups', '--num-competitors', '--rating-periods', '--repeats', '--warmup', '--models', '--seed']


def run(args, **kwargs):
    print(f'+ {" ".join(str(a) for a in args)}')
    return subprocess.run(args, check=True, **kwargs)


def bench_ref(ref, repo_root, bench_script, out_path, bench_args, tmp_dir):
    if ref == '.':
        run(['uv', 'run', 'python', str(bench_script), '-o', str(out_path), *bench_args], cwd=repo_root)
        return
    worktree = Path(tmp_dir) / f'worktree_{ref.replace("/", "_")}'
    run(['git', 'worktree', 'add', '--detach', str(worktree), ref], cwd=repo_root)
    try:
        run(['uv', 'run', 'python', str(bench_script), '-o', str(out_path), *bench_args], cwd=worktree)
    finally:
        run(['git', 'worktree', 'remove', '--force', str(worktree)], cwd=repo_root)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--base', default='main', help='baseline git ref (default: main)')
    parser.add_argument('--head', default='.', help="candidate git ref, or '.' for the current working tree")
    for arg in BENCH_ARGS:
        parser.add_argument(arg, default=None)
    args = parser.parse_args()

    bench_args = []
    for arg in BENCH_ARGS:
        value = getattr(args, arg.lstrip('-').replace('-', '_'))
        if value is not None:
            bench_args.extend([arg, value])

    repo_root = Path(
        subprocess.run(
            ['git', 'rev-parse', '--show-toplevel'], capture_output=True, text=True, check=True
        ).stdout.strip()
    )
    bench_script = repo_root / 'benchmarks' / 'bench.py'
    compare_script = repo_root / 'benchmarks' / 'compare.py'

    with tempfile.TemporaryDirectory(prefix='riix_ab_') as tmp_dir:
        base_json = Path(tmp_dir) / 'base.json'
        head_json = Path(tmp_dir) / 'head.json'
        print(f'\n=== benchmarking base: {args.base} ===')
        bench_ref(args.base, repo_root, bench_script, base_json, bench_args, tmp_dir)
        print(f'\n=== benchmarking head: {args.head} ===')
        bench_ref(args.head, repo_root, bench_script, head_json, bench_args, tmp_dir)
        print('\n=== comparison (speedup = base / head, >1 means head is faster) ===')
        run([sys.executable, str(compare_script), str(base_json), str(head_json)])


if __name__ == '__main__':
    main()
