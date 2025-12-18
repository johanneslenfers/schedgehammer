# Only needed since this is in the same repo as schedgehammer.
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
##############################################################

import catbench as cb
from interop_problem import problem_from_study

from schedgehammer.benchmark import benchmark
from schedgehammer.random_search import RandomSearch
from schedgehammer.genetic_tuner import GeneticTuner
from schedgehammer.tuner import EvalBudget

ITERATIONS = 1000
REPETITIONS = 10
BENCHMARKS = [
    # "spmm",
    "spmv",
    # "sddmm",
    "mttkrp",
    # "ttv",
    # "asum",
    "harris",
    # "kmeans",
    # "stencil",
]

if __name__ == "__main__":
    constrained_tuners = {
        "GeneticTuner with constraints": GeneticTuner(),
        # "GeneticTuner with constraints and LocalMutation": GeneticTuner(
        #     local_mutation=True
        # ),
        "RandomSearch with constraints": RandomSearch(),
    }

    tuners = {
        "GeneticTuner without constraints": GeneticTuner(),
        "GeneticTuner with LocalMutation": GeneticTuner(local_mutation=True),
        "RandomSearch without constraints": RandomSearch(),
    }

    for benchmark_name in BENCHMARKS:
        study = cb.benchmark(benchmark_name)
        problem = problem_from_study(study)
        benchmark(
            problem,
            [EvalBudget(ITERATIONS)],
            constrained_tuners,
            f"results_catbench/{benchmark_name}",
            REPETITIONS,
            export_raw_data=True,
        )