import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional, Tuple

from schedgehammer.param_types import ParamValue
from schedgehammer.problem import Problem

ParameterConfiguration = Dict[str, ParamValue]


class Budget(ABC):
    @abstractmethod
    def in_budget(self) -> bool:
        pass


@dataclass
class EvalBudget(Budget):
    max_evaluations: int
    num_evaluations: int = 0

    def in_budget(self) -> bool:
        return self.num_evaluations <= self.max_evaluations


@dataclass
class TimeBudget(Budget):
    seconds: float
    started: Optional[datetime] = None

    def in_budget(self) -> bool:
        if self.started is None:
            self.started = datetime.now()
            return True
        else:
            return (datetime.now() - self.started).total_seconds() <= self.seconds


class Tuner(ABC):
    problem: Problem
    budget: Budget
    best_score: float = math.inf
    best_config: ParameterConfiguration = None

    def __init__(self, problem: Problem, budget: Budget):
        self.problem = problem
        self.budget = budget

    def log_state(self):
        print("\033[H\033[J", end="")
        for name, value in self.best_config.items():
            print(f">>> {name}:", value)
        print("Score:", self.best_score)

    def evaluate_config(self, config) -> float:
        score = self.problem.cost_function(config)

        if isinstance(self.budget, EvalBudget):
            self.budget.num_evaluations += 1

        if score < self.best_score:
            self.best_score = score
            self.best_config = config
        return score

    @abstractmethod
    def tune(self) -> Tuple[ParameterConfiguration, float]:
        raise NotImplementedError
