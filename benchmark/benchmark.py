# Only needed since this is in the same repo as schedgehammer.
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
##############################################################


import json
import os
from collections import defaultdict
import catbench as cb
from interop_problem import problem_from_study

from schedgehammer.genetic_tuner import GeneticTuner
from schedgehammer.problem import Problem
from schedgehammer.tuner import EvalBudget

ITERATIONS = 3000
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

        study = cb.benchmark(benchmark_name)
        problem = problem_from_study(study)
        result = GeneticTuner().tune(problem, [EvalBudget(ITERATIONS)])
        results[benchmark_name] = result.best_score_list()
    min_benchmark_iterations = min(
        [len(results[benchmark_name]) for benchmark_name in BENCHMARKS]
    )  # Hotfix for genetic tuner running only 90% of stated iterations
    results["total_time_schedgehammer"] = [0] * min_benchmark_iterations
    for benchmark_name in BENCHMARKS:
        for iteration in range(min_benchmark_iterations):
            results["total_time_schedgehammer"][iteration] += results[benchmark_name][
                iteration
            ]

    with open(os.path.join("results", "results.json"), "w") as f:
        f.write(json.dumps(results))
