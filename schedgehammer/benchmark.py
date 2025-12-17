import csv
import itertools
import json
import multiprocessing
import os
from collections import defaultdict
from datetime import datetime
from multiprocessing import pool
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np

from schedgehammer.param_types import ParamValue
from schedgehammer.problem import Problem
from schedgehammer.result import EvaluationResult
from schedgehammer.tuner import Budget, Tuner


def statistical_description(a):
    return {"avg": np.mean(a), "std": np.std(a)}


class NoDaemonProcess(multiprocessing.Process):
    @property
    def daemon(self):
        return False

    @daemon.setter
    def daemon(self, value):
        pass


class NoDaemonContext(type(multiprocessing.get_context())):
    Process = NoDaemonProcess


# We sub-class multiprocessing.pool.Pool instead of multiprocessing.Pool
# because the latter is only a wrapper function, not a proper class.
class NestablePool(pool.Pool):
    def __init__(self, *args, **kwargs):
        kwargs["context"] = NoDaemonContext()
        super(NestablePool, self).__init__(*args, **kwargs)


class ArchivedResult:
    runs_of_tuners: dict[str, list[list[EvaluationResult]]]

    def __init__(self, runs_of_tuners: dict[str, list[list[EvaluationResult]]] = None):
        if runs_of_tuners is None:
            runs_of_tuners = {}
        self.runs_of_tuners = runs_of_tuners

    def load_runs(self, folder: str, tuner_names: list = None):
        path = Path(folder)
        if not path.glob("*.csv"):
            print("No csv files found in directory")
            return
        tuners_in_results = set([x.name.split("-")[0] for x in path.glob("*.csv")])
        for tuner in tuners_in_results:
            if tuner_names is not None and tuner not in tuner_names:
                continue
            self.runs_of_tuners[tuner] = []
            for file in path.glob(f"{tuner}*.csv"):
                self.runs_of_tuners[tuner].append([])
                with open(file, newline="") as csvfile:
                    reader = csv.reader(csvfile)
                    header = next(reader)
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

    def delete(self, tuner_name):
        del self.runs_of_tuners[tuner_name]

    def rename(self, old_tuner_name, new_tuner_name):
        self.runs_of_tuners[new_tuner_name] = self.runs_of_tuners.pop(old_tuner_name)

    @staticmethod
    def _csv_value_to_param_type(value: str) -> ParamValue:
        # Returns bools as ints, would need to go through whole dataset to detect bools
        try:
            return json.loads(value)
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

    def plot(self, output_file) -> None:
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
        plt.yscale("log")
        plt.legend()
        plt.savefig(output_file)


def run_task(obj):
    (
        (problem, budget, export_raw_data, output_path, timeout_secs),
        (tuner_name, tuner),
        i,
    ) = obj
    print(f"begin {tuner_name}-{i}", flush=True)
    # Instantiate problem if it's a class (needed for multiprocessing)
    if isinstance(problem, type):
        problem = problem()
    result = tuner.tune(problem, budget, timeout_secs)
    if export_raw_data:
        result.generate_csv(os.path.join(output_path, f"runs/{tuner_name}-{i}.csv"))
    print(f"end {tuner_name}-{i}", flush=True)


def benchmark(
    problem,
    budget: list[Budget],
    tuner_list: dict[str, Tuner],
    output_path: str = "",
    repetitions: int = 1,
    export_raw_data: bool = False,
    parallel: int = 1,
    timeout_secs: float = 90,
):
    Path(output_path).mkdir(parents=True, exist_ok=True)

    tasks = list(
        itertools.product(
            [(problem, budget, export_raw_data, output_path, timeout_secs)],
            tuner_list.items(),
            range(repetitions),
        )
    )
    # Avoid multiprocessing when parallel == 1 to prevent pickling issues
    # with locally defined lambdas inside constraint expressions.
    if parallel == 1:
        for t in tasks:
            run_task(t)
    else:
        with NestablePool(parallel) as pool:
            pool.map(run_task, tasks)
