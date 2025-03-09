from dataclasses import dataclass

from schedgehammer.tuner import Tuner, TuningAttempt


@dataclass
class ScheduleRandomSearch(Tuner):
    def do_tuning(self, tuning_attempt: TuningAttempt):
        while tuning_attempt.in_budget():
            config = {}
            for param_name, param in tuning_attempt.problem.params.items():
                config[param_name] = param.choose_random()

            tuning_attempt.evaluate_config(config)
