import random
from dataclasses import dataclass

from schedgehammer.schedule_type import ScheduleParam, SchedulePlanningTree
from schedgehammer.tuner import Tuner, TuningAttempt


@dataclass
class SchedulesGeneticTuner(Tuner):
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
        while attempt.in_budget(self.schedule_local_mutation_budget_share):
            ret = {}
            while True:
                for [name, param] in attempt.problem.params.items():
                    ret[name] = param.choose_random()

                if attempt.fulfills_all_constraints(ret):
                    break

            cost = attempt.evaluate_config(ret)
            population.append((ret, schedule_param.last_generated_tree, cost))
        elitism_size = int(len(population) * self.elitism_share)
        elites = sorted(population, key=lambda x: x[2])[:elitism_size]
        while attempt.in_budget():
            ret, tree, old_cost = random.choice(elites)
            tree.randomly_tweak_primitive_params()
            cost = attempt.evaluate_config(ret)
            if cost < old_cost:
                print(
                    "\033[92mMutation improved cost from",
                    old_cost,
                    "to",
                    cost,
                    "\033[0m",
                )
            else:
                print(
                    "\033[93mMutation increased cost from",
                    old_cost,
                    "to",
                    cost,
                    "\033[0m",
                )
