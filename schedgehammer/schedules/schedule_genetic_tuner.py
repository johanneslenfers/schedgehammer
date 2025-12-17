import random
from dataclasses import dataclass

from schedgehammer.schedules.schedule_type import ScheduleParam
from schedgehammer.tuner import Tuner, TuningAttempt


@dataclass
class ScheduleGeneticTuner(Tuner):
    population_size: int = 100
    elitism_share: float = 0.05
    reproduction_share: float = 0.3
    crossover_prob: float = 0.5
    mutation_prob: float = 0.1
    local_mutation: bool = False
    schedule_local_mutation_budget_share: float = 0.2

    def do_tuning(self, attempt: TuningAttempt):
        population = []
        schedule_params = [
            (name, param)
            for name, param in attempt.problem.params.items()
            if isinstance(param, ScheduleParam)
        ]

        if len(schedule_params) != 1:
            raise Exception("This tuning method should only be used with one schedule")

        schedule_param_name, schedule_param = schedule_params[0]
        print(f"Genetic tuner: Building initial population (target: {self.population_size})...", flush=True)
        pop_count = 0
        while attempt.in_budget(self.schedule_local_mutation_budget_share):
            ret = {}
            while True:
                for [name, param] in attempt.problem.params.items():
                    ret[name] = param.choose_random()

                if attempt.fulfills_all_constraints(ret):
                    break

            cost = attempt.evaluate_config(ret)
            population.append((ret, schedule_param.last_generated_tree, cost))
            pop_count += 1
            if pop_count % 10 == 0 or pop_count == self.population_size:
                print(f"Genetic tuner: Population {pop_count}/{self.population_size}, "
                      f"best so far: {min(p[2] for p in population):.6f}", flush=True)
        elitism_size = int(len(population) * self.elitism_share)
        elites = sorted(population, key=lambda x: x[2])[:elitism_size]
        print(f"Genetic tuner: Initialized population of {len(population)}, "
              f"elites={elitism_size}, best={elites[0][2]:.6f}", flush=True)
        
        mutation_count = 0
        while attempt.in_budget():
            ret, tree, old_cost = random.choice(elites)
            tree.randomly_tweak_primitive_params()
            cost = attempt.evaluate_config(ret)
            mutation_count += 1
            
            # Print progress every 10 mutations or on improvement
            if mutation_count % 10 == 0 or cost < old_cost:
                budget_used = attempt.current_evaluation
                budget_total = attempt.budgets[0].max_evaluations if hasattr(attempt.budgets[0], 'max_evaluations') else '?'
                if cost < old_cost:
                    print(
                        f"\033[92mGenetic: eval {budget_used}/{budget_total}, "
                        f"mutation improved {old_cost:.6f} -> {cost:.6f}, best={attempt.best_score:.6f}\033[0m",
                        flush=True
                    )
                else:
                    print(
                        f"Genetic: eval {budget_used}/{budget_total}, "
                        f"mutation {old_cost:.6f} -> {cost:.6f}, best={attempt.best_score:.6f}",
                        flush=True
                    )
