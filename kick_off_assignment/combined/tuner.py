import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable

from param_types import ParamValueTypes

from kick_off_assignment.combined.problem_config import ProblemConfig


@dataclass
class Tuner(ABC):
    problem_config: ProblemConfig
    cost_function: Callable[[dict[str, ParamValueTypes]], float]
    time_budget: float
    latest_score: float = None

    def log_state(self):
        print("\033[H\033[J", end="")
        for name, param in self.problem_config.params.items():
            print(f">>> {name}:", param.val)
        print("Score:", self.latest_score)

    def evaluate_config(self):
        self.latest_score = self.cost_function(self.problem_config.to_dict())
        return self.latest_score

    @abstractmethod
    def tune(self):
        raise NotImplementedError


class LocalSearchTuner(Tuner):
    @abstractmethod
    def make_iteration(self):
        raise NotImplementedError

    def tune(self) -> float:
        start_time = time.time()
        while time.time() - start_time < self.time_budget:
            #TODO