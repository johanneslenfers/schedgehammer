# Only needed since this is in the same repo as schedgehammer.
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
##############################################################

from pathlib import Path
import csv

import catbench as cb
from pyatf import TP, Interval, Set, Tuner, abort_conditions
from pyatf.tuning_data import Cost, CostFunctionError
from pyatf.search_techniques import AUCBandit

ITERATIONS = 1000
REPETITIONS = 10
OUTPUT_DIR = "results_catbench/pyatf_harris"

# Harris parameters
exp_1_1024 = Set(1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)
exp_vec = Set(2, 4)

# Constraints encoded directly at TP construction (non-redundant)
tuned_ls0 = TP('tuned_ls0', exp_1_1024,
               lambda tuned_ls0, tuned_ls1: (tuned_ls0 * tuned_ls1) <= 1024
            )
tuned_ls1 = TP('tuned_ls1', exp_1_1024)

tuned_gs0 = TP('tuned_gs0', exp_1_1024,
               lambda tuned_gs0, tuned_ls0: tuned_gs0 % tuned_ls0 == 0
            )
tuned_gs1 = TP('tuned_gs1', exp_1_1024)

tuned_tileX = TP('tuned_tileX', Interval(1, 1024),
                lambda tuned_tileX, tuned_tileY: (
                    (tuned_tileX * tuned_tileY) <= 1024 and
                    (tuned_tileY == 1 or tuned_tileY % 2 == 0) and
                    (tuned_tileY != 1 or ((tuned_tileX != 1024) and (tuned_tileX != 1022)))
                )
            )
tuned_tileY = TP('tuned_tileY', Interval(1, 1024),
                )
tuned_vec = TP('tuned_vec', exp_vec,
               lambda tuned_vec, tuned_tileX: (tuned_tileX + 4) % tuned_vec == 0
            )


# Load catbench study
study = cb.benchmark("harris")

# Fidelity parameters (defaults from rise_fidelity_params)
fidelity_params = {
    'iterations': 10,
    'repeats': 5,
    'wait_between_repeats': 0,
    'wait_after_run': 10,
}


def cost_function(config):
    # transformed_config = {
    #     'tuned_ls0': config['tuned_ls0'],
    #     'tuned_ls1': config['tuned_ls1'],
    #     'tuned_gs0': config['tuned_gs0'],
    #     'tuned_gs1': config['tuned_gs1'],
    #     'tuned_tileX': config['tuned_tileX'],
    #     'tuned_tileY': config['tuned_tileY'],
    #     'tuned_vec': config['tuned_vec'],
    # }
    # print(f"Testing configuration: {transformed_config}")
    try:
        query_result = study.query(config, fidelity_params)
        if "compute_time" not in query_result:
            print(f"WARNING: 'compute_time' not in result. Keys: {query_result.keys()}")
            # Fallback: return a high penalty cost
            raise CostFunctionError(f"No compute_time in result. Got keys: {list(query_result.keys())}")
        result = query_result["compute_time"]
        print(f"Resulting compute_time: {result}")
        return Cost(result)
    except Exception as e:
        print(f"ERROR in cost function: {e}")
        print(f"  Config: {config}")
        print(f"  Exception type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
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
    print(f"Benchmarking: Harris with pyATF")
    print(f"Iterations: {ITERATIONS}, Repetitions: {REPETITIONS}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"{'='*60}\n")

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    tuning_params = [
        tuned_ls0,
        tuned_ls1,
        tuned_gs0,
        tuned_gs1,
        tuned_tileX,
        tuned_tileY,
        tuned_vec,
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

        csv_path = os.path.join(OUTPUT_DIR, f"AUCBandit-Harris-{rep}.csv")
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
