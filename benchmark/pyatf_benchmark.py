



# Only needed since this is in the same repo as schedgehammer.
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
##############################################################

import catbench as cb
from interop_problem import pyatf_from_study

from pyatf import Tuner, abort_conditions
from pyatf.search_techniques import AUCBandit, Random

ITERATIONS = 100
BENCHMARKS = [
    "mttkrp",
    # "spmv",
    # "harris",
    # Add more benchmarks as needed
]

if __name__ == "__main__":
    for benchmark_name in BENCHMARKS:
        print(f"\n{'='*60}")
        print(f"Benchmarking: {benchmark_name}")
        print(f"{'='*60}\n")
        
        # Load catbench study
        study = cb.benchmark(benchmark_name)
        
        # Convert to pyATF format
        try:
            tuning_params, cost_function = pyatf_from_study(study)
        except NotImplementedError as e:
            print(f"Skipping {benchmark_name}: {e}")
            continue
        
        print(f"Tuning parameters: {[tp.name for tp in tuning_params]}")
        print(f"Running {ITERATIONS} iterations with AUCBandit search...\n")
        
        # Run pyATF tuning with AUCBandit (recommended default)
        tuning_result = Tuner().tuning_parameters(*tuning_params) \
                               .search_technique(AUCBandit()) \
                               .tune(cost_function, abort_conditions.Evaluations(ITERATIONS))
        
        print(f"\nBest configuration found:")
        print(f"  Cost: {tuning_result.costs[tuning_result.best_configuration]}")
        print(f"  Configuration: {tuning_result.best_configuration}")
        
        # Optional: Compare with random search
        print(f"\nRunning {ITERATIONS} iterations with Random search for comparison...\n")
        tuning_result_random = Tuner().tuning_parameters(*tuning_params) \
                                      .search_technique(Random()) \
                                      .tune(cost_function, abort_conditions.Evaluations(ITERATIONS))
        
        print(f"\nRandom search best configuration:")
        print(f"  Cost: {tuning_result_random.costs[tuning_result_random.best_configuration]}")
        print(f"  Configuration: {tuning_result_random.best_configuration}")
        
        print(f"\nAUCBandit vs Random improvement: "
              f"{tuning_result_random.costs[tuning_result_random.best_configuration] / tuning_result.costs[tuning_result.best_configuration]:.2f}x")

