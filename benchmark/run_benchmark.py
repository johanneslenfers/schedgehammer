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

from multiprocessing import Pool

ITERATIONS = 1000
REPETITIONS = 15
BENCHMARKS = [
    "harris",
    "spmm",
    "spmv",
    "sddmm",
    "mttkrp",
    "ttv",
    # "asum",
    "kmeans",
    "stencil",
]


def run_benchmark(benchmark_name):
    constrained_tuners = {
        "GeneticTuner with constraints": GeneticTuner(),
        "GeneticTuner with constraints and LocalMutation": GeneticTuner(
            local_mutation=True
        ),
        "RandomSearch with constraints": RandomSearch(),
    }

    tuners = {
        "GeneticTuner without constraints": GeneticTuner(),
        "GeneticTuner with LocalMutation": GeneticTuner(local_mutation=True),
        "RandomSearch without constraints": RandomSearch(),
    }

    study = cb.benchmark(benchmark_name)
    problem = problem_from_study(study)
    benchmark(
        problem,
        [EvalBudget(ITERATIONS)],
        constrained_tuners,
        f"results/9802a88/{benchmark_name}/constrained",
        REPETITIONS,
        export_raw_data=True,
    )

    # remove constraints
    problem.constraints = []
    benchmark(
        problem,
        [EvalBudget(ITERATIONS)],
        tuners,
        f"results/9802a88/{benchmark_name}/unconstrained",
        REPETITIONS,
        export_raw_data=True,
    )

if __name__ == "__main__":
    with Pool(3) as p:
        p.map(run_benchmark, BENCHMARKS)

