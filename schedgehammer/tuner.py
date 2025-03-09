import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict

from schedgehammer.param_types import ParamValue
from schedgehammer.problem import Problem
from schedgehammer.result import EvaluationResult, TuningResult
from schedgehammer.constraint import Solver

ParameterConfiguration = Dict[str, ParamValue]


class Budget(ABC):
    @abstractmethod
    def in_budget(self, tuner, reserve=0) -> bool:
        pass


@dataclass
class EvalBudget(Budget):
    max_evaluations: int

    def in_budget(self, tuner, reserve=0) -> bool:
        return tuner.current_evaluation <= self.max_evaluations * (1 - reserve)


@dataclass
class TimeBudget(Budget):
    seconds: float

    def in_budget(self, tuner, reserve=0) -> bool:
        return time.perf_counter() - tuner.start_time <= self.seconds * (1 - reserve)


class TuningAttempt:
    problem: Problem
    solver: Solver
    budgets: list[Budget]
    record_of_evaluations: list[EvaluationResult]
    start_time: float
    current_evaluation: int = 0

    last_improvement_evaluation: int = 0
    last_improvement_time: float

    best_score: float = math.inf
    best_config: ParameterConfiguration = None

    evaluation_cumulative_duration: float = 0

    def __init__(self, problem: Problem, budgets: list[Budget]):
        self.problem = problem
        self.solver = self.problem.get_solver()
        self.budgets = budgets
        self.record_of_evaluations = []
        self.start_time = time.perf_counter()
        self.last_improvement_time = time.perf_counter()

    def evaluate_config(self, config: ParameterConfiguration) -> float:
        if not self.in_budget():
            raise Exception("Budget spent!")

        translated_config = self.translate_config_for_evaluation(config)

        start = time.perf_counter()
        score = self.problem.cost_function(translated_config)
        self.evaluation_cumulative_duration += time.perf_counter() - start

        self.record_of_evaluations.append(
            EvaluationResult(
                score,
                list(config.values()),
                self.current_evaluation,
                time.perf_counter() - self.start_time,
            )
        )
        self.current_evaluation += 1

        if score < self.best_score:
            self.best_score = score
            self.best_config = config
        return score

    def translate_config_for_evaluation(self, config: ParameterConfiguration) -> ParameterConfiguration:
        new_config = {}
        for param_name, param in self.problem.params.items():
            new_config[param_name] = param.translate_for_evaluation(config[param_name])
        return new_config

    def fulfills_all_constraints(self, config: ParameterConfiguration) -> bool:
        for constraint in self.problem.constraints:
            if not constraint.evaluate(config):
                return False
        return True

    def create_result(self):
        complete_execution_time = time.perf_counter() - self.start_time
        return TuningResult(
            self.problem.params,
            self.record_of_evaluations,
            complete_execution_time,
            complete_execution_time - self.evaluation_cumulative_duration,
            self.evaluation_cumulative_duration,
        )

    def in_budget(self, reserve=0) -> bool:
        return all([b.in_budget(self, reserve) for b in self.budgets])

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
