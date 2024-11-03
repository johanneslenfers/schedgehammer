import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Tuple

from schedgehammer.param_types import ParamValue
from schedgehammer.problem import Problem
from schedgehammer.result import EvaluationResult, TuningResult

ParameterConfiguration = Dict[str, ParamValue]


class Budget(ABC):
    @abstractmethod
    def in_budget(self, tuner) -> bool:
        pass

@dataclass
class EvalBudget(Budget):
    max_evaluations: int

    def in_budget(self, tuner) -> bool:
        return tuner.num_evaluations <= self.max_evaluations


@dataclass
class TimeBudget(Budget):
    seconds: float

    def in_budget(self, tuner) -> bool:
        return (datetime.now() - tuner.start_time).total_seconds() <= self.seconds


class Tuner(ABC):
    problem: Problem
    budget: Budget
    num_evaluations: int = 0
    start_time: datetime = datetime.now()
    record_of_evaluations: list[EvaluationResult] = []

    def __init__(self, problem: Problem, budget: Budget):
        self.problem = problem
        self.budget = budget

    def log_state(self):
        print("\033[H\033[J", end="")
        if len(self.record_of_evaluations) == 0:
            print("No evaluations recorded")
        else:
            best_eval = self.record_of_evaluations[-1]
            for name, value in zip(self.problem.params.keys(), best_eval.config):
                print(f">>> {name}:", value)
            print("Score:", best_eval.score)

    def best_score(self):
        if len(self.record_of_evaluations) == 0:
            return math.inf
        else:
            return self.record_of_evaluations[-1].score

    def evaluate_config(self, config: ParameterConfiguration) -> float:
        score = self.problem.cost_function(config)

        if score < self.best_score():
            self.record_of_evaluations.append(
                EvaluationResult(score, list(config.values()), self.num_evaluations, datetime.now() - self.start_time)
            )

        self.num_evaluations += 1
        return score

    def create_result(self):
        return TuningResult(self.problem.params, self.record_of_evaluations)

    @abstractmethod
    def tune(self) -> TuningResult:
        raise NotImplementedError
