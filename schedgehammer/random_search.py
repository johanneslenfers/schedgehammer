import random
from dataclasses import dataclass

from schedgehammer.tuner import Tuner, TuningAttempt


@dataclass
class RandomSearch(Tuner):
    check_constraints: bool = True

    def do_tuning(self, tuning_attempt: TuningAttempt):
        while tuning_attempt.in_budget():
            config = next(tuning_attempt.solver.solve())
            tuning_attempt.evaluate_config(config)
