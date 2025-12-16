# Only needed since this is in the same repo as schedgehammer.
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
##############################################################

import catbench as cb
from pyatf import TP, Interval, Set, Tuner, abort_conditions
from pyatf.tuning_data import Cost, CostFunctionError
from pyatf.search_techniques import AUCBandit, Random
import itertools
from pathlib import Path
import csv

ITERATIONS = 100
REPETITIONS = 5
OUTPUT_DIR = "results/pyatf_mttkrp"

# Define MTKKRP tuning parameters based on catbench definition
# IntExponential parameters: chunk_size, unroll_factor, omp_chunk_size
# Using explicit power-of-2 values: 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024

# chunk_size must be defined first (no constraint)
chunk_size = TP(
    'chunk_size',
    Set(1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)  # 2^0 to 2^10
)

# unroll_factor depends on chunk_size
unroll_factor = TP(
    'unroll_factor',
    Set(1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024),  # 2^0 to 2^10
    lambda unroll_factor, chunk_size: unroll_factor < chunk_size
)

omp_chunk_size = TP(
    'omp_chunk_size',
    Set(1, 2, 4, 8, 16, 32, 64, 128, 256)  # 2^0 to 2^8
)

# Common parameters from taco_common_parameters
omp_num_threads = TP(
    'omp_num_threads',
    Interval(2, 20)  # Integer(bounds=(2, 20), default=16)
)

omp_scheduling_type = TP(
    'omp_scheduling_type',
    Set(0, 1, 2)  # Categorical(categories=[0, 1, 2], default=0)
)

omp_monotonic = TP(
    'omp_monotonic',
    Set(0, 1)  # Categorical(categories=[0, 1], default=0)
)

omp_dynamic = TP(
    'omp_dynamic',
    Set(0, 1)  # Categorical(categories=[0, 1], default=0)
)

# Permutation of length 5: (1, 0, 2, 3, 4) default
all_permutations = list(itertools.permutations([0, 1, 2, 3, 4]))
all_permutations = [list(p) for p in all_permutations]  # Convert to lists
permutation = TP(
    'permutation',
    Set(*all_permutations)
)

# Load catbench study
study = cb.benchmark("mttkrp")

# Fidelity parameters (use defaults from taco_fidelity_params)
fidelity_params = {
    'iterations': 10,
    'repeats': 5,
    'wait_between_repeats': 0,
    'wait_after_run': 10
}

# Cost function
def cost_function(config):
    """Cost function that queries catbench with the configuration."""
    # Config already has explicit values, just convert permutation to string
    transformed_config = {
        'chunk_size': config['chunk_size'],
        'unroll_factor': config['unroll_factor'],
        'omp_chunk_size': config['omp_chunk_size'],
        'omp_num_threads': config['omp_num_threads'],
        'omp_scheduling_type': config['omp_scheduling_type'],
        'omp_monotonic': config['omp_monotonic'],
        'omp_dynamic': config['omp_dynamic'],
        'permutation': str(config['permutation'])  # Convert list to string for catbench
    }
    
    try:
        result = study.query(transformed_config, fidelity_params)["compute_time"]
        return Cost(result)
    except Exception as e:
        # If query fails, raise CostFunctionError to penalize this configuration
        raise CostFunctionError(f"Configuration failed: {e}")


def save_tuning_data_to_csv(tuning_data, output_path):
    """Save tuning data history to CSV file."""
    history = tuning_data.history
    
    # Check if history is empty
    if history.is_empty():
        print(f"Warning: No history to save to {output_path}")
        return
    
    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Get parameter names from first entry
        first_entry = next(iter(history))
        param_names = list(first_entry.configuration.keys())
        
        # Write header
        writer.writerow(['evaluation', 'cost', 'timestamp'] + param_names)
        
        # Write data rows
        for entry in history:
            row = [
                entry.evaluations,
                entry.cost,
                entry.timestamp
            ] + [entry.configuration[param] for param in param_names]
            writer.writerow(row)


if __name__ == "__main__":
    print(f"{'='*60}")
    print(f"Benchmarking: MTTKRP with pyATF")
    print(f"Iterations: {ITERATIONS}, Repetitions: {REPETITIONS}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"{'='*60}\n")
    
    # Create output directory
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
    
    # Store results for statistics
    aucbandit_costs = []
    random_costs = []
    
    # Run multiple repetitions
    for rep in range(REPETITIONS):
        print(f"{'='*60}")
        print(f"Repetition {rep + 1}/{REPETITIONS}")
        print(f"{'='*60}\n")
        
        # Run pyATF tuning with AUCBandit
        print(f"Running AUCBandit search ({ITERATIONS} iterations)...")
        best_config_aucbandit, min_cost_aucbandit, tuning_data_aucbandit = \
            Tuner().tuning_parameters(*tuning_params) \
                   .search_technique(AUCBandit()) \
                   .tune(cost_function, abort_conditions.Evaluations(ITERATIONS))
        
        aucbandit_costs.append(min_cost_aucbandit)
        
        # Save AUCBandit results to CSV
        aucbandit_csv = os.path.join(OUTPUT_DIR, f"AUCBandit-{rep}.csv")
        save_tuning_data_to_csv(tuning_data_aucbandit, aucbandit_csv)
        print(f"Saved: {aucbandit_csv}")
        
        # # Run Random search for comparison
        # print(f"\nRunning Random search ({ITERATIONS} iterations)...")
        # best_config_random, min_cost_random, tuning_data_random = \
        #     Tuner().tuning_parameters(*tuning_params) \
        #            .search_technique(Random()) \
        #            .tune(cost_function, abort_conditions.Evaluations(ITERATIONS))
        
        # random_costs.append(min_cost_random)
        
        # Save Random results to CSV
        # random_csv = os.path.join(OUTPUT_DIR, f"Random-{rep}.csv")
        # save_tuning_data_to_csv(tuning_data_random, random_csv)
        # print(f"Saved: {random_csv}")
        
        # print(f"\nRepetition {rep + 1} Summary:")
        # print(f"  AUCBandit best cost: {min_cost_aucbandit:.4f}")
        # print(f"  Random best cost: {min_cost_random:.4f}")
        # print(f"  Improvement: {min_cost_random / min_cost_aucbandit:.2f}x\n")
    
    # Print overall statistics
    print(f"\n{'='*60}")
    print(f"Overall Results ({REPETITIONS} repetitions)")
    print(f"{'='*60}")
    print(f"\nAUCBandit:")
    print(f"  Mean: {sum(aucbandit_costs) / len(aucbandit_costs):.4f}")
    print(f"  Min:  {min(aucbandit_costs):.4f}")
    print(f"  Max:  {max(aucbandit_costs):.4f}")
   
    # avg_improvement = (sum(random_costs) / len(random_costs)) / (sum(aucbandit_costs) / len(aucbandit_costs))
    # print(f"\nAverage improvement: {avg_improvement:.2f}x")
    print(f"\nAll results saved to: {OUTPUT_DIR}/")
