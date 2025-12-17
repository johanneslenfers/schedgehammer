# Only needed since this is in the same repo as schedgehammer.
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
##############################################################

import csv
from pathlib import Path

import catbench as cb

# OpenTuner imports
import opentuner
from opentuner import MeasurementInterface
from opentuner import Result
from opentuner.tuningrunmain import TuningRunMain
from opentuner.search.manipulator import ConfigurationManipulator
from opentuner.search.manipulator import IntegerParameter, EnumParameter, PermutationParameter

# Tuning config
ITERATIONS = 1000
REPETITIONS = 2
OUTPUT_DIR = "results_catbench/opentuner_spmv"
INVALID_COST = 1e12

# Load catbench study
study = cb.benchmark("spmv")

# Fidelity parameters (defaults)
fidelity_params = {
    'iterations': 10,
    'repeats': 5,
    'wait_between_repeats': 0,
    'wait_after_run': 10
}


class SpmvOpenTuner(MeasurementInterface):
    def __init__(self, args):
        # OpenTuner's MeasurementInterface expects CLI args in constructor
        super().__init__(args)
        self._history = []  # collect per-evaluation results
        self._eval_counter = 0

    def manipulator(self):
        m = ConfigurationManipulator()
        # Exponential sets as explicit values
        exp_values_10 = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
        exp_values_8 = [1, 2, 4, 8, 16, 32, 64, 128, 256]

        # Three chunk sizes for SpMV (as per dataset columns)
        m.add_parameter(EnumParameter('chunk_size', exp_values_10))
        m.add_parameter(EnumParameter('chunk_size2', exp_values_10))
        m.add_parameter(EnumParameter('chunk_size3', exp_values_10))

        m.add_parameter(EnumParameter('omp_chunk_size', exp_values_8))
        m.add_parameter(IntegerParameter('omp_num_threads', 2, 20))
        m.add_parameter(EnumParameter('omp_scheduling_type', [0, 1, 2]))
        m.add_parameter(EnumParameter('omp_monotonic', [0, 1]))
        m.add_parameter(EnumParameter('omp_dynamic', [0, 1]))

        # PermutationParameter of length 5
        m.add_parameter(PermutationParameter('permutation', list(range(5))))
        return m

    def run(self, desired_result, input, limit=None):
        cfg = desired_result.configuration.data

        # Constraint: permutation[4] must equal 4; penalize otherwise
        try:
            perm = cfg['permutation']
            if not (isinstance(perm, (list, tuple)) and len(perm) == 5 and perm[4] == 4):
                return Result(time=INVALID_COST)
        except Exception:
            return Result(time=INVALID_COST)

        transformed_config = {
            'chunk_size': cfg['chunk_size'],
            'chunk_size2': cfg['chunk_size2'],
            'chunk_size3': cfg['chunk_size3'],
            'omp_chunk_size': cfg['omp_chunk_size'],
            'omp_num_threads': cfg['omp_num_threads'],
            'omp_scheduling_type': cfg['omp_scheduling_type'],
            'omp_monotonic': cfg['omp_monotonic'],
            'omp_dynamic': cfg['omp_dynamic'],
            'permutation': str(cfg['permutation'])
        }

        try:
            cost = float(study.query(transformed_config, fidelity_params)["compute_time"])
        except Exception:
            cost = INVALID_COST

        # record history entry
        self._eval_counter += 1
        self._history.append({
            'evaluation': self._eval_counter,
            'cost': cost,
            'config': transformed_config,
        })

        return Result(time=cost)

    def save_results_csv(self, tuning_run, output_csv):
        Path(os.path.dirname(output_csv)).mkdir(parents=True, exist_ok=True)
        with open(output_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            # header: evaluation, cost + params (derive from first history entry)
            if self._history:
                param_names = list(self._history[0]['config'].keys())
            else:
                param_names = []
            writer.writerow(['evaluation', 'cost'] + param_names)

            # dump collected history entries
            for entry in self._history:
                cfg = entry['config']
                row = [entry['evaluation'], entry['cost']] + [cfg.get(n) for n in param_names]
                writer.writerow(row)


def run_repetitions():
    print('=' * 60)
    print(f'Benchmarking: SPMV with OpenTuner')
    print(f'Iterations: {ITERATIONS}, Repetitions: {REPETITIONS}')
    print(f'Output directory: {OUTPUT_DIR}')
    print('=' * 60)

    min_costs = []
    for rep in range(REPETITIONS):
        print('\n' + ('-' * 60))
        print(f'Repetition {rep + 1}/{REPETITIONS}')
        print('-' * 60)

        # Build OpenTuner argument parser and parse defaults
        parser = opentuner.default_argparser()
        # Parse defaults without CLI input; users can override when running directly
        args = parser.parse_args([])
        # Ensure fixed number of evaluations (evaluations/test_limit)
        args.test_limit = ITERATIONS

        interface = SpmvOpenTuner(args)
        # Run tuning with OpenTuner's main runner
        TuningRunMain(interface, args).main()

        # Save CSV
        csv_path = os.path.join(OUTPUT_DIR, f'OpenTuner-SPMV-{rep}.csv')
        interface.save_results_csv(None, csv_path)
        print(f'Saved: {csv_path}')

        # Compute best cost from collected history
        if interface._history:
            best_entry = min(interface._history, key=lambda e: e['cost'])
            min_costs.append(best_entry['cost'])
            print('Best configuration (from collected history):')
            for k, v in best_entry['config'].items():
                print(f'  {k} = {v}')
            print(f"Min cost: {best_entry['cost']}")
        else:
            print('No results recorded.')

    print('\n' + ('=' * 60))
    print(f'Overall Results ({REPETITIONS} repetitions)')
    print('=' * 60)
    if min_costs:
        print(f'  Mean: {sum(min_costs) / len(min_costs):.4f}')
        print(f'  Min:  {min(min_costs):.4f}')
        print(f'  Max:  {max(min_costs):.4f}')
    else:
        print('  No results recorded.')
    print(f"\nAll results saved to: {OUTPUT_DIR}/")


if __name__ == '__main__':
    run_repetitions()
