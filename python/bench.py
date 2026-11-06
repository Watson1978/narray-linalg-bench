"""Times one problem at one size in one dtype and prints a single TSV line.

The Python counterpart of script/bench.rb, with the same columns:

  impl  backend  problem  size  dtype  ms/step  best_seconds  all_seconds

  GPU=1 DTYPE=fp32 python/.venv/bin/python python/bench.py --problem gaussian_process --size mid --inner 20
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import problems  # noqa: E402
from backend import backend_name, describe, dtype_name, synchronize  # noqa: E402


def measure(step, rounds, warmup, inner):
    for _ in range(warmup):
        for _ in range(inner):
            step()
    synchronize()
    times = []
    for _ in range(rounds):
        started = time.perf_counter()
        for _ in range(inner):
            step()
        synchronize()
        times.append(time.perf_counter() - started)
    return times


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem", required=True)
    parser.add_argument("--size", default="small")
    parser.add_argument("--variant")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--inner", type=int, default=1)
    args = parser.parse_args()

    problem = problems.fetch(args.problem)
    step = problem.bench_step(problem.fixture(args.size), args.variant)
    times = measure(step, args.rounds, args.warmup, args.inner)
    best = min(times)
    name = f"{args.problem}:{args.variant}" if args.variant else args.problem
    print("\t".join([backend_name(), describe(), name, args.size, dtype_name(),
                     f"{best / args.inner * 1000:.4f}", f"{best:.6f}",
                     ",".join(f"{t:.6f}" for t in times)]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
