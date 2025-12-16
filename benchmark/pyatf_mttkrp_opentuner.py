# Only needed since this is in the same repo as schedgehammer.
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
##############################################################

import catbench as cb
from pyatf import TP, Interval, Set, Tuner, abort_conditions
from pyatf.tuning_data import Cost, CostFunctionError
from pyatf.search_techniques import OpenTuner
from pathlib import Path
import csv
import itertools

ITERATIONS = 100
REPETITIONS = 5
OUTPUT_DIR = "results/pyatf_mttkrp_opentuner"

# Define MTTKRP tuning parameters (no constraints here; we'll enforce in cost function)
chunk_size = TP('chunk_size', Set(1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024))
unroll_factor = TP('unroll_factor', Set(1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024))
omp_chunk_size = TP('omp_chunk_size', Set(1, 2, 4, 8, 16, 32, 64, 128, 256))

omp_num_threads = TP('omp_num_threads', Interval(2, 20))
omp_scheduling_type = TP('omp_scheduling_type', Set(0, 1, 2))
omp_monotonic = TP('omp_monotonic', Set(0, 1))
omp_dynamic = TP('omp_dynamic', Set(0, 1))

all_permutations = [list(p) for p in itertools.permutations([0, 1, 2, 3, 4])]
permutation = TP('permutation', Set(*all_permutations))

# Load catbench study
study = cb.benchmark("mttkrp")

# Fidelity parameters (use defaults)
fidelity_params = {
    'iterations': 10,
    'repeats': 5,
    'wait_between_repeats': 0,
    'wait_after_run': 10
}

# Cost function with manual constraint checking (OpenTuner lacks constraints)
# Constraint to enforce: unroll_factor < chunk_size
# Return a large cost if invalid to penalize.
INVALID_COST = 1e12

def cost_function(config):
    # Manual constraint check
    if not (config['unroll_factor'] < config['chunk_size']):
        return Cost(INVALID_COST)

    transformed_config = {
        'chunk_size': config['chunk_size'],
        'unroll_factor': config['unroll_factor'],
        'omp_chunk_size': config['omp_chunk_size'],
        'omp_num_threads': config['omp_num_threads'],
        'omp_scheduling_type': config['omp_scheduling_type'],
        'omp_monotonic': config['omp_monotonic'],
        'omp_dynamic': config['omp_dynamic'],
        'permutation': str(config['permutation'])
    }

    try:
        result = study.query(transformed_config, fidelity_params)["compute_time"]
        return Cost(result)
    except Exception as e:
        # Penalize failures
        return Cost(INVALID_COST)


def save_tuning_data_to_csv(tuning_data, output_path):
    history = tuning_data.history
    if history.is_empty():
        print(f"Warning: No history to save to {output_path}")
        return
    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        first_entry = next(iter(history))
        param_names = list(first_entry.configuration.keys())
        writer.writerow(['evaluation', 'cost', 'timestamp'] + param_names)
        for entry in history:
            row = [
                entry.evaluations,
                entry.cost,
                entry.timestamp
            ] + [entry.configuration[param] for param in param_names]
            writer.writerow(row)


if __name__ == "__main__":
    print(f"{'='*60}")
    print(f"Benchmarking: MTTKRP with pyATF OpenTuner")
    print(f"Iterations: {ITERATIONS}, Repetitions: {REPETITIONS}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"{'='*60}\n")

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    tuning_params = [
        chunk_size,
        unroll_factor,
        omp_chunk_size,
        omp_num_threads,
        omp_scheduling_type,
        omp_monotonic,
        omp_dynamic,
        permutation
    ]

    print(f"Tuning parameters: {[tp.name for tp in tuning_params]}\n")

    min_costs = []

    for rep in range(REPETITIONS):
        print(f"{'='*60}")
        print(f"Repetition {rep + 1}/{REPETITIONS}")
        print(f"{'='*60}\n")

        best_config, min_cost, tuning_data = \
            Tuner().tuning_parameters(*tuning_params) \
                   .search_technique(OpenTuner()) \
                   .tune(cost_function, abort_conditions.Evaluations(ITERATIONS))

        min_costs.append(min_cost)

        csv_path = os.path.join(OUTPUT_DIR, f"OpenTuner-{rep}.csv")
        save_tuning_data_to_csv(tuning_data, csv_path)
        print(f"Saved: {csv_path}")

        print(f"\nRepetition {rep + 1} Summary:")
        print(f"  Best cost: {min_cost:.4f}")
        print(f"  Best configuration:")
        for k, v in best_config.items():
            print(f"    {k}: {v}")
        print()

    print(f"\n{'='*60}")
    print(f"Overall Results ({REPETITIONS} repetitions)")
    print(f"{'='*60}")
    print(f"  Mean: {sum(min_costs) / len(min_costs):.4f}")
    print(f"  Min:  {min(min_costs):.4f}")
    print(f"  Max:  {max(min_costs):.4f}")
    print(f"\nAll results saved to: {OUTPUT_DIR}/")
