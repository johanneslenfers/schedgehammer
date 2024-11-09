# Only needed since this is in the same repo as schedgehammer.
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
##############################################################

import catbench as cb
from interop_problem import problem_from_study

from schedgehammer.benchmark import benchmark
from schedgehammer.random_search import RandomSearch
from schedgehammer.genetic_tuner import GeneticTuner
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

    tuners = [GeneticTuner(), RandomSearch()]

    for benchmark_name in BENCHMARKS:
        study = cb.benchmark(benchmark_name)
        problem = problem_from_study(study)
        benchmark(problem, [EvalBudget(ITERATIONS)], tuners, f'results/{benchmark_name}', 10, export_raw_data=True)
