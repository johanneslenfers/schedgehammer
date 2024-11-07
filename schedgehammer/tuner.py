import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Dict, Tuple, Optional

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
        return tuner.current_evaluation <= self.max_evaluations


@dataclass
class TimeBudget(Budget):
    seconds: float

    def in_budget(self, tuner) -> bool:
        return (datetime.now() - tuner.start_time).total_seconds() <= self.seconds


class TuningAttempt:
    problem: Problem
    budgets: list[Budget]
    record_of_evaluations: list[EvaluationResult]
    start_time: datetime
    current_evaluation: int = 0
    best_score: float = math.inf
    best_config: ParameterConfiguration = None

    def __init__(self, problem: Problem, budgets: list[Budget]):
        self.problem = problem
        self.budgets = budgets
        self.record_of_evaluations = []
        self.start_time = datetime.now()

    def evaluate_config(self, config: ParameterConfiguration) -> float:
        score = self.problem.cost_function(config)

        self.record_of_evaluations.append(
            EvaluationResult(
                score,
                list(config.values()),
                self.current_evaluation,
                datetime.now() - self.start_time,
            )
        )
        self.current_evaluation += 1

        if score < self.best_score:
            self.best_score = score
            self.best_config = config

        return score

    def create_result(self):
        return TuningResult(self.problem.params, self.record_of_evaluations)

    def in_budget(self) -> bool:
        return all([b.in_budget(self) for b in self.budgets])

    def log_state(self):
        print("\033[H\033[J", end="")
        for name, value in self.best_config.items():
            print(f">>> {name}:", value)
        print("Score:", self.best_score)


class Tuner(ABC):

    def tune(self, problem: Problem, budgets: list[Budget]) -> TuningResult:
        attempt = TuningAttempt(problem, budgets)
        self.do_tuning(attempt)
        return attempt.create_result()

    @abstractmethod
    def do_tuning(self, tuning_attempt: TuningAttempt):
        raise NotImplementedError
