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

ITERATIONS = 500
BENCHMARKS = [
    "harris",
]

if __name__ == "__main__":

    tuners = {
        "GeneticTuner": GeneticTuner(),
    }

    for benchmark_name in BENCHMARKS:
        study = cb.benchmark(benchmark_name)
        problem = problem_from_study(study)
        result = benchmark(problem, [EvalBudget(ITERATIONS)], tuners, f'results/{benchmark_name}', 5, export_raw_data=True)
        result.plot(f"results/{benchmark_name}.png")
