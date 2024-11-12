import csv
import json
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.ticker import LinearLocator

from schedgehammer.param_types import ParamValue
from schedgehammer.problem import Problem
from schedgehammer.result import EvaluationResult
from schedgehammer.tuner import Budget, Tuner


def statistical_description(a):
    return {"avg": np.mean(a), "std": np.std(a)}


class ArchivedResult:
    param_names: list[str]
    runs_of_tuners: dict[list[list[EvaluationResult]]]

    def __init__(self, dir: Path):
        assert dir.glob("*.csv"), "No csv files found in directory"
        self.runs_of_tuners = defaultdict(list)
        tuners_in_results = set([x.name.split("-")[0] for x in dir.glob("*")])
        for tuner in tuners_in_results:
            for file in dir.glob(f"{tuner}*.csv"):
                self.runs_of_tuners[tuner].append([])
                with open(file, newline="") as csvfile:
                    reader = csv.reader(csvfile)
                    header = next(reader)
                    if not hasattr(self, "param_names"):
                        self.param_names = header[3:]
                    for row in reader:
                        self.runs_of_tuners[tuner][-1].append(
                            EvaluationResult(
                                score=float(row[1]),
                                config=[
                                    self._csv_value_to_param_type(x) for x in row[3:]
                                ],
                                num_evaluation=int(row[0]),
                                timestamp=float(row[2]),
                            )
                        )

    @staticmethod
    def _csv_value_to_param_type(value: str) -> ParamValue:
        # Returns bools as ints, would need to go through whole dataset to detect bools
        try:
            json.loads(value)
        except json.JSONDecodeError:
            # Regular string
            return

    @staticmethod
    def _get_best_scores(runs) -> list[list[float]]:
        best_scores = []
        for run in runs:
            best_scores.append([])
            best_score = float("inf")
            for record in run:
                if record.score < best_score:
                    best_score = record.score
                best_scores[-1].append(best_score)
        return best_scores

    def plot(self) -> None:
        plt.figure()
        for tuner, runs in self.runs_of_tuners.items():
            zipped = list(zip(*self._get_best_scores(runs)))
            median = []
            upper_bound = []
            lower_bound = []
            xs = []
            for i in range(len(zipped)):
                xs.append(i)
                median.append(np.median(zipped[i]))
                lower_bound.append(np.percentile(zipped[i], 50 - 68.27 / 2))
                upper_bound.append(np.percentile(zipped[i], 50 + 68.27 / 2))
            plt.plot(xs, median, label=tuner)
            plt.fill_between(xs, lower_bound, upper_bound, alpha=0.3)

        plt.xlabel("function evaluations")
        plt.ylabel("cost")
        plt.legend()
        plt.show()


def benchmark(
    problem: Problem,
    budget: list[Budget],
    tuner_list: dict[str, Tuner],
    output_path: str = "",
    repetitions: int = 1,
    export_raw_data: bool = False,
):
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
                result.generate_csv(
                    os.path.join(output_path, f"runs/{tuner_name}-{i}.csv")
                )

            score_list = result.best_score_list()
            results.append(score_list)

            total_time.append(result.complete_execution_time)
            algorithm_time.append(result.algorithm_execution_time)
            final_score.append(score_list[-1])

        report[tuner_name] = {
            "tuner_desc": str(tuner),
            "starttime": str(start_date),
            "repetitions": repetitions,
            "execution_time": statistical_description(total_time),
            "algorithm_time": statistical_description(algorithm_time),
            "final_score": statistical_description(final_score),
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

    plt.xlabel("function evaluations")
    plt.ylabel("cost")
    plt.yscale("log")
    plt.legend()
    plt.savefig(os.path.join(output_path, "plot.png"))

    with open(os.path.join(output_path, "report.json"), "w") as f:
        f.write(json.dumps(report, indent=2))
