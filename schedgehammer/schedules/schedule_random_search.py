from dataclasses import dataclass

from schedgehammer.tuner import Tuner, TuningAttempt


@dataclass
class ScheduleRandomSearch(Tuner):
    def do_tuning(self, tuning_attempt: TuningAttempt):
        evaluation_count = 0
        while tuning_attempt.in_budget():
            config = {}
            for param_name, param in tuning_attempt.problem.params.items():
                config[param_name] = param.choose_random()

            best_before = tuning_attempt.best_score
            cost = tuning_attempt.evaluate_config(config)
            evaluation_count += 1
            is_improvement = cost < best_before
            
            # Print progress every 10 evaluations or on improvement
            if evaluation_count % 10 == 0 or is_improvement:
                budget_used = tuning_attempt.current_evaluation
                budget_total = tuning_attempt.budgets[0].max_evaluations if hasattr(tuning_attempt.budgets[0], 'max_evaluations') else '?'
                if is_improvement:
                    print(
                        f"\033[92mRandom search: eval {budget_used}/{budget_total}, "
                        f"IMPROVED {best_before:.6f} -> {cost:.6f}\033[0m",
                        flush=True
                    )
                else:
                    print(
                        f"Random search: eval {budget_used}/{budget_total}, "
                        f"cost={cost:.6f}, best={tuning_attempt.best_score:.6f}",
                        flush=True
                    )
