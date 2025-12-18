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
from opentuner.search.manipulator import IntegerParameter, EnumParameter

# Tuning config
ITERATIONS = 1000
REPETITIONS = 10
OUTPUT_DIR = "results/results_catbench/opentuner_harris"
INVALID_COST = 1e12

# Load catbench study
study = cb.benchmark("harris")

# Fidelity parameters (defaults from rise_fidelity_params)
fidelity_params = {
    'iterations': 10,
    'repeats': 5,
    'wait_between_repeats': 0,
    'wait_after_run': 10,
}


class HarrisOpenTuner(MeasurementInterface):
    def __init__(self, args):
        super().__init__(args)
        self._history = []
        self._eval_counter = 0

    def manipulator(self):
        m = ConfigurationManipulator()
        # IntExponential (powers of 2 within bounds)
        exp_1_1024 = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
        exp_vec = [2, 4]

        m.add_parameter(EnumParameter('tuned_ls0', exp_1_1024))
        m.add_parameter(EnumParameter('tuned_ls1', exp_1_1024))
        m.add_parameter(EnumParameter('tuned_gs0', exp_1_1024))
        m.add_parameter(EnumParameter('tuned_gs1', exp_1_1024))
        m.add_parameter(IntegerParameter('tuned_tileX', 1, 1024))
        m.add_parameter(IntegerParameter('tuned_tileY', 1, 1024))
        m.add_parameter(EnumParameter('tuned_vec', exp_vec))
        return m

    def _constraints_ok(self, cfg):
        try:
            if cfg['tuned_gs0'] % cfg['tuned_ls0'] != 0:
                return False
            if cfg['tuned_gs1'] % cfg['tuned_ls1'] != 0:
                return False
            if (cfg['tuned_tileX'] + 4) % cfg['tuned_vec'] != 0:
                return False
            if (cfg['tuned_tileX'] * cfg['tuned_tileY']) > 1024:
                return False
            if (cfg['tuned_ls0'] * cfg['tuned_ls1']) > 1024:
                return False
            if not (cfg['tuned_tileX'] == 1 or cfg['tuned_tileX'] % 2 == 0):
                return False
            if not (cfg['tuned_tileY'] == 1 or cfg['tuned_tileY'] % 2 == 0):
                return False
            if not (cfg['tuned_tileY'] != 1 or ((cfg['tuned_tileX'] != 1024) and (cfg['tuned_tileX'] != 1022))):
                return False
        except Exception:
            return False
        return True

    def run(self, desired_result, input, limit=None):
        cfg = desired_result.configuration.data

        # Enforce constraints; penalize if violated
        if not self._constraints_ok(cfg):
            return Result(time=INVALID_COST)

        transformed_config = {
            'tuned_ls0': cfg['tuned_ls0'],
            'tuned_ls1': cfg['tuned_ls1'],
            'tuned_gs0': cfg['tuned_gs0'],
            'tuned_gs1': cfg['tuned_gs1'],
            'tuned_tileX': cfg['tuned_tileX'],
            'tuned_tileY': cfg['tuned_tileY'],
            'tuned_vec': cfg['tuned_vec'],
        }

        try:
            cost = float(study.query(transformed_config, fidelity_params)["compute_time"])
        except Exception:
            cost = INVALID_COST

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
            param_names = list(self._history[0]['config'].keys()) if self._history else []
            writer.writerow(['evaluation', 'cost'] + param_names)
            for entry in self._history:
                cfg = entry['config']
                row = [entry['evaluation'], entry['cost']] + [cfg.get(n) for n in param_names]
                writer.writerow(row)


def run_repetitions():
    print('=' * 60)
    print(f'Benchmarking: Harris with OpenTuner')
    print(f'Iterations: {ITERATIONS}, Repetitions: {REPETITIONS}')
    print(f'Output directory: {OUTPUT_DIR}')
    print('=' * 60)

    min_costs = []
    for rep in range(REPETITIONS):
        print('\n' + ('-' * 60))
        print(f'Repetition {rep + 1}/{REPETITIONS}')
        print('-' * 60)

        parser = opentuner.default_argparser()
        args = parser.parse_args([])
        args.test_limit = ITERATIONS

        interface = HarrisOpenTuner(args)
        TuningRunMain(interface, args).main()

        csv_path = os.path.join(OUTPUT_DIR, f'OpenTuner-Harris-{rep}.csv')
        interface.save_results_csv(None, csv_path)
        print(f'Saved: {csv_path}')

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
