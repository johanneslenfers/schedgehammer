import json
import os
from datetime import datetime
from pathlib import Path

from schedgehammer.problem import Problem
from schedgehammer.tuner import Tuner, Budget
import numpy as np
import matplotlib.pyplot as plt

def statistical_description(a):
    return {
        'avg': np.mean(a),
        'std': np.std(a)
    }

def benchmark(problem: Problem,
              budget: list[Budget],
              tuner_list: dict[str, Tuner],
              output_path: str = '',
              repetitions: int = 1,
              export_raw_data: bool = False):

    report = {}

    Path(output_path).mkdir(parents=True, exist_ok=True)

    plt.figure()

    for tuner_name, tuner in tuner_list.items():
        total_time = []
        algorithm_time = []
        final_score = []
        start_date = datetime.now()

        results = []
        for i in range(repetitions):
            result = tuner.tune(problem, budget)
            if export_raw_data:
                result.generate_csv(os.path.join(output_path, f'runs/{tuner_name}-{i}.csv'))

            score_list = result.best_score_list()
            results.append(score_list)

            total_time.append(result.complete_execution_time)
            algorithm_time.append(result.algorithm_execution_time)
            final_score.append(score_list[-1])

        report[tuner_name] = {
            'tuner_desc': str(tuner),
            'starttime': str(start_date),
            'repetitions': repetitions,
            'execution_time': statistical_description(total_time),
            'algorithm_time': statistical_description(algorithm_time),
            'final_score': statistical_description(final_score)
        }

        results = list(zip(*results))
        median = []
        upper_bound = []
        lower_bound = []
        xs = []
        for i in range(len(results)):
            xs.append(i)
            median.append(np.median(results[i]))
            lower_bound.append(np.percentile(results[i], 50 - 68.27 / 2))
            upper_bound.append(np.percentile(results[i], 50 + 68.27 / 2))

        plt.plot(xs, median, label=tuner_name)
        plt.fill_between(xs, lower_bound, upper_bound, alpha=0.3)

    plt.xlabel('function evaluations')
    plt.ylabel('cost')
    plt.yscale('log')
    plt.legend()
    plt.savefig(os.path.join(output_path, 'plot.png'))

    with open(os.path.join(output_path, 'report.json'), 'w') as f:
        f.write(json.dumps(report, indent=2))
