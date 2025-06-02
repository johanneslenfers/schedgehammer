# Only needed since this is in the same repo as schedgehammer.
import sys
import os

from schedgehammer.genetic_tuner_2 import GeneticTuner2
from schedgehammer.random_search_2 import RandomSearch2

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
##############################################################

import catbench as cb
from interop_problem import problem_from_study

from schedgehammer.benchmark import benchmark
from schedgehammer.tuner import EvalBudget

ITERATIONS = 1000
REPETITIONS = 50
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
    constrained_tuners = {
        "GeneticTuner2": GeneticTuner2(),
        "RandomSearch2": RandomSearch2(),
    }

    for benchmark_name in BENCHMARKS:
        study = cb.benchmark(benchmark_name)
        problem = problem_from_study(study)
        benchmark(
            problem,
            [EvalBudget(ITERATIONS)],
            constrained_tuners,
            f"results/catbench/{benchmark_name}",
            REPETITIONS,
            export_raw_data=True,
        )
