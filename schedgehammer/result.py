import csv
import math
from dataclasses import dataclass
from datetime import timedelta
import matplotlib.pyplot as plt
from pathlib import Path

from schedgehammer.param_types import ParamValue, Param

@dataclass
class EvaluationResult:
    score: float
    config: list[ParamValue]
    num_evaluation: int
    timestamp: timedelta

@dataclass
class TuningResult:
    parameters: dict[str, Param]
    record_of_evaluations: list[EvaluationResult]

    def generate_csv(self, name="evaluations.csv", only_improvements=False):
        Path(name).parent.mkdir(parents=True, exist_ok=True)
        with open(name, 'w', newline='') as file:
            writer = csv.writer(file)
            # Header
            writer.writerow(["num_evaluation", "score", "timestamp"] + list(self.parameters.keys()))
            # Body
            best_score = math.inf
            for record in self.record_of_evaluations:
                if record.score < best_score or not only_improvements:
                    writer.writerow([record.num_evaluation, record.score, record.timestamp.total_seconds()] +
                                    record.config)
                    best_score = record.score

    def generate_plot(self, name="plot.png"):
        xs = []
        ys = []
        best_score = math.inf

        for records in self.record_of_evaluations:
            if records.score < best_score:
                xs.append(records.num_evaluation)
                ys.append(records.score)
                best_score = records.score

        plt.plot(xs, ys, label=name)

        plt.xlabel('function evaluations')
        plt.ylabel('cost')
        plt.savefig(name)
