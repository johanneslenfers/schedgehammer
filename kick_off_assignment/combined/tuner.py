import math
import time
from abc import ABC, abstractmethod
from typing import Tuple, Dict

from param_types import ParamValue
from problem import Problem

ParameterConfiguration = Dict[str, ParamValue]

class Tuner(ABC):
    problem: Problem

    best_score: float = math.inf
    best_config: ParameterConfiguration = None

    budget_evaluations: int

    def __init__(self, problem: Problem, budget_evaluations: int):  # TODO do we also want to use time budget?
        self.problem = problem
        self.budget_evaluations = budget_evaluations

    def log_state(self):
        print("\033[H\033[J", end="")
        for name, param in self.problem_config.params.items():
            print(f">>> {name}:", param.val)
        print("Score:", self.latest_score)

    def evaluate_config(self, config):
        score = self.problem.cost_function(config)
        if score < self.best_score:
            self.best_score = score
            self.best_config = config
        return score

    @abstractmethod
    def tune(self) -> Tuple[Problem, float]:
        raise NotImplementedError
