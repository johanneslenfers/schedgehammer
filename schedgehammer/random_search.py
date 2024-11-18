from dataclasses import dataclass

from schedgehammer.constraint import Solver
from schedgehammer.tuner import Tuner, TuningAttempt

@dataclass
class RandomSearch(Tuner):
    check_constraints: bool = True

    def do_tuning(self, attempt: TuningAttempt):

        solver = Solver(
            {k: v.get_value_range() for k, v in attempt.problem.params.items()},
            [c.to_constraint() for c in attempt.problem.constraints]
        )
        while attempt.in_budget():
            config = solver.solve()
            attempt.evaluate_config(config)
