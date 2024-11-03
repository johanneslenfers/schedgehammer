import csv
from dataclasses import dataclass
from datetime import timedelta
import matplotlib.pyplot as plt

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

    def generate_csv(self, name="evaluations.csv"):
        with open(name, 'w', newline='') as file:
            writer = csv.writer(file)
            # Header
            writer.writerow(["num_evaluation", "score", "timestamp"] + list(self.parameters.keys()))
            # Body
            for record in self.record_of_evaluations:
                writer.writerow([record.num_evaluation, record.score, record.timestamp.total_seconds()] +
                                record.config)

    def generate_plot(self, name="plot.png"):
        xs = []
        ys = []

        for records in self.record_of_evaluations:
            xs.append(records.num_evaluation)
            ys.append(records.score)

        plt.plot(xs, ys, label=name)

        plt.xlabel('function evaluations')
        plt.ylabel('cost')
        plt.savefig(name)
