import json
import os
from collections import defaultdict

import catbench as cb
from schedgehammer.genetic_tuner import GeneticTuner
from schedgehammer.problem import Problem
from schedgehammer.tuner import EvalBudget

ITERATIONS = 1000
BENCHMARKS = [
    "spmm",
    "spmv",
    "sddmm",
    "mttkrp",
    "ttv",
    "asum",
    "harris",
    "kmeans",
    "stencil",
]

if __name__ == "__main__":
    results = defaultdict(list)

    for benchmark_name in BENCHMARKS:

        def score_callback(config, score):
            results[benchmark_name].append(score)

        study = cb.benchmark(benchmark_name)
        problem = Problem.from_interopt_format(study)
        _ = GeneticTuner(problem, [EvalBudget(ITERATIONS)], score_callback).tune()
    min_benchmark_iterations = min(
        [len(results[benchmark_name]) for benchmark_name in BENCHMARKS]
    )  # Hotfix for genetic tuner running only 900 iterations
    results["total_time_schedgehammer"] = [0] * min_benchmark_iterations
    for benchmark_name in BENCHMARKS:
        for iteration in range(min_benchmark_iterations):
            results["total_time_schedgehammer"][iteration] += results[benchmark_name][
                iteration
            ]
