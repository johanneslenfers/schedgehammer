from dataclasses import dataclass

from schedgehammer.tuner import Tuner, TuningAttempt

@dataclass
class RandomSearch(Tuner):
    check_constraints: bool = True

    def do_tuning(self, attempt: TuningAttempt):
        while attempt.in_budget():
            config = {}
            while True:
                for [name, param] in attempt.problem.params.items():
                    config[name] = param.choose_random()

                if not self.check_constraints or attempt.fulfills_all_constraints(config):
                    break

            attempt.evaluate_config(config)
