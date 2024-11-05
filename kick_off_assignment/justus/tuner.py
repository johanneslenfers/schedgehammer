import math
from abc import ABCMeta, abstractmethod
from typing import Generator, Tuple

from tuning_problem import TuningProblem, TuningConfig


class Tuner(metaclass=ABCMeta):
    tuning_problem: TuningProblem

    def __init__(self, tuning_problem: TuningProblem):
        self.tuning_problem = tuning_problem

    def print(self, solution: TuningConfig, score: float, rewrite: bool = False):
        s = f"Best score: {score}\nConfig:"
        for name in solution:
            s += f"\n  {name}: {str(solution[name])}"
        if rewrite:
            num_lines = s.count("\n") + 1
            print(f"\033[1A\x1b[2K" * num_lines, end="")
        print(s)

    def forever(self):
        """
        Tune forever and display best result.
        """
        try:
            tune = self.tune()
            (solution, score) = next(tune)
            self.print(solution, score)
            last_score = score
            while True:
                (solution, score) = next(tune)
                if last_score != score:
                    self.print(solution, score, rewrite=True)
                    last_score = score
        except KeyboardInterrupt:
            return

    def num_evaluations(self, n):
        """
        Tune for n evaluations.
        :param n:
        """
        tune = self.tune()
        (solution, score) = next(tune)
        self.print(solution, score)
        last_score = score
        for i in range(n - 1):
            (solution, score) = next(tune)
            if last_score != score:
                self.print(solution, score, rewrite=True)
                last_score = score

    def tune(self) -> Generator[Tuple[TuningConfig, float], None, None]:
        tune = self.generate_tuning()
        best_score = math.inf
        best_solution = None
        while True:
            solution, score = next(tune)
            if score < best_score:
                best_score = score
                best_solution = solution.copy()
            yield best_solution, best_score

    @abstractmethod
    def generate_tuning(self) -> Generator[Tuple[TuningConfig, float], None, None]:
        pass

    def random_config(self) -> TuningConfig:
        c = {}
        for name in self.tuning_problem.parameters:
            c[name] = self.tuning_problem.parameters[name].random_value()
        return c
