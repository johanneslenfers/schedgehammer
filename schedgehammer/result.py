import csv
import math
from dataclasses import dataclass
from pathlib import Path

from schedgehammer.param_types import Param, ParamValue
from schedgehammer.problem import Problem


@dataclass
class EvaluationResult:
    score: float
    config: list[ParamValue]
    num_evaluation: int
    timestamp: float


# @dataclass
class TuningResult:

    def __init__(self,
                parameters: dict[str, Param],
                record_of_evaluations: list[EvaluationResult],
                complete_execution_time: float,
                algorithm_execution_time: float,
                evaluation_execution_time: float,
                ) -> None:
        self.parameters: dict[str, Param] = parameters
        self.record_of_evaluations: list[EvaluationResult] = record_of_evaluations
        self.complete_execution_time: float = complete_execution_time
        self.algorithm_execution_time: float = algorithm_execution_time
        self.evaluation_execution_time: float = evaluation_execution_time

    def generate_csv(self, name="evaluations.csv", only_improvements=False):
        Path(name).parent.mkdir(parents=True, exist_ok=True)
        with open(name, "w", newline="") as file:
            writer = csv.writer(file)
            # Header
            writer.writerow(
                ["num_evaluation", "score", "timestamp"] + list(self.parameters.keys())
            )
            # Body
            best_score = math.inf
            for record in self.record_of_evaluations:
                if record.score < best_score or not only_improvements:
                    writer.writerow(
                        [record.num_evaluation, record.score, record.timestamp]
                        + record.config
                    )
                    best_score = record.score

    def best_score_list(self) -> list[float]:
        l = []
        best = math.inf
        for record in self.record_of_evaluations:
            if record.score < best:
                best = record.score
            l.append(best)
        return l
