# Only needed since this is in the same repo as schedgehammer.
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
##############################################################

import itertools
from pathlib import Path
import csv

import catbench as cb
from pyatf import TP, Interval, Set, Tuner, abort_conditions
from pyatf.tuning_data import Cost, CostFunctionError
from pyatf.search_techniques import AUCBandit

ITERATIONS = 1000
REPETITIONS = 10
OUTPUT_DIR = "results/results_catbench/pyatf_spmv"

# Define SPMV tuning parameters based on dataset columns
# chunk_size, chunk_size2, chunk_size3 as powers of 2 up to 1024
chunk_size = TP('chunk_size', Set(1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024))
chunk_size2 = TP('chunk_size2', Set(1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024))
chunk_size3 = TP('chunk_size3', Set(1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024))

omp_chunk_size = TP('omp_chunk_size', Set(1, 2, 4, 8, 16, 32, 64, 128, 256))
omp_num_threads = TP('omp_num_threads', Interval(2, 20))
omp_scheduling_type = TP('omp_scheduling_type', Set(0, 1, 2))
omp_monotonic = TP('omp_monotonic', Set(0, 1))
omp_dynamic = TP('omp_dynamic', Set(0, 1))

# Permutation of length 5
all_permutations = list(itertools.permutations([0, 1, 2, 3, 4]))
all_permutations = [list(p) for p in all_permutations]
permutation = TP('permutation', Set(*all_permutations))

# Load catbench study
study = cb.benchmark("spmv")

# Fidelity parameters (defaults)
fidelity_params = {
    'iterations': 10,
    'repeats': 5,
    'wait_between_repeats': 0,
    'wait_after_run': 10
}


def cost_function(config):
    # Constraint: permutation[4] must equal 4; penalize otherwise
    try:
        perm = config['permutation']
        if not (isinstance(perm, (list, tuple)) and len(perm) == 5 and perm[4] == 4):
            raise CostFunctionError("Invalid configuration: permutation[4] must equal 4")
    except Exception as e:
        raise CostFunctionError(f"Invalid configuration (permutation constraint): {e}")

    transformed_config = {
        'chunk_size': config['chunk_size'],
        'chunk_size2': config['chunk_size2'],
        'chunk_size3': config['chunk_size3'],
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
        raise CostFunctionError(f"Configuration failed: {e}")


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
            row = [entry.evaluations, entry.cost, entry.timestamp] + [entry.configuration[p] for p in param_names]
            writer.writerow(row)


if __name__ == "__main__":
    print(f"{'='*60}")
    print(f"Benchmarking: SPMV with pyATF")
    print(f"Iterations: {ITERATIONS}, Repetitions: {REPETITIONS}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"{'='*60}\n")

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    tuning_params = [
        chunk_size,
        chunk_size2,
        chunk_size3,
        omp_chunk_size,
        omp_num_threads,
        omp_scheduling_type,
        omp_monotonic,
        omp_dynamic,
        permutation,
    ]

    costs = []

    for rep in range(REPETITIONS):
        print(f"{'='*60}")
        print(f"Repetition {rep + 1}/{REPETITIONS}")
        print(f"{'='*60}\n")

        print(f"Running AUCBandit search ({ITERATIONS} iterations)...")
        best_config, min_cost, tuning_data = (
            Tuner()
            .tuning_parameters(*tuning_params)
            .search_technique(AUCBandit())
            .tune(cost_function, abort_conditions.Evaluations(ITERATIONS))
        )
        costs.append(min_cost)

        csv_path = os.path.join(OUTPUT_DIR, f"AUCBandit-SPMV-{rep}.csv")
        save_tuning_data_to_csv(tuning_data, csv_path)
        print(f"Saved: {csv_path}")

    print(f"\n{'='*60}")
    print(f"Overall Results ({REPETITIONS} repetitions)")
    print(f"{'='*60}")
    if costs:
        print(f"  Mean: {sum(costs) / len(costs):.4f}")
        print(f"  Min:  {min(costs):.4f}")
        print(f"  Max:  {max(costs):.4f}")
    else:
        print("  No results recorded.")
    print(f"\nAll results saved to: {OUTPUT_DIR}/")
