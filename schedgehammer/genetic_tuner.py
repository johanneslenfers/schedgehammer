import random
from dataclasses import dataclass

from schedgehammer.meta_tree import MetaTree
from schedgehammer.schedule_type import ScheduleParam, ScheduleTree
from schedgehammer.tuner import Tuner, TuningAttempt


@dataclass
class GeneticTuner(Tuner):
    check_constraints: bool = True
    population_size: int = 100
    elitism_share: float = 0.2
    reproduction_share: float = 0.3
    crossover_prob: float = 0.5
    mutation_prob: float = 0.1
    local_mutation: bool = False
    schedule_local_mutation_budget_share: float = 0.2

    def do_tuning(self, attempt: TuningAttempt):
        elitism_size = int(self.population_size * self.elitism_share)
        reproduction_size = int(self.population_size * self.reproduction_share)

        population = []
        schedule_params = [
            (name, param)
            for name, param in attempt.problem.params.items()
            if isinstance(param, ScheduleParam)
        ]
        if len(schedule_params) == 1:
            schedule_param_name, schedule_param = schedule_params[0]
            meta_tree = MetaTree(schedule_param)
            while attempt.in_budget(self.schedule_local_mutation_budget_share):
                ret = {}
                while True:
                    for [name, param] in attempt.problem.params.items():
                        ret[name] = param.choose_random()

                    if not self.check_constraints or attempt.fulfills_all_constraints(
                        ret
                    ):
                        break

                cost = attempt.evaluate_config(ret)
                schedule_param.last_generated_tree.add_to_meta_tree(meta_tree, cost)
                population.append((ret, schedule_param.last_generated_tree, cost))
            best_meta_nodes = meta_tree.get_n_best_children(6)  # TODO: Not hardcoded
            for node in best_meta_nodes:
                print(str(node))
                print([op.name for op in node.get_geneaology()])
                print("-" * 100)
            while attempt.in_budget():
                ret = {}
                while True:
                    for [name, param] in attempt.problem.params.items():
                        if name == schedule_param_name:
                            tree = schedule_param.create_schedule()
                            geneaology_to_apply = random.choice(
                                best_meta_nodes
                            ).get_geneaology()
                            for operation in geneaology_to_apply:
                                _, _, method = schedule_param.try_appending_method(
                                    tree, [operation]
                                )
                                if not method:
                                    print(
                                        f"\033[93mReplicating {operation.name} failed\033[0m"
                                    )
                                    break
                        ret[name] = param.choose_random()

                    if not self.check_constraints or attempt.fulfills_all_constraints(
                        ret
                    ):
                        break
                cost = attempt.evaluate_config(ret)
                print("COST IN 2nd phase", cost)
        elif len(schedule_params) == 0:
            for _ in range(self.population_size):
                ret = {}
                while True:
                    for [name, param] in attempt.problem.params.items():
                        ret[name] = param.choose_random()

                    if not self.check_constraints or attempt.fulfills_all_constraints(
                        ret
                    ):
                        break

                cost = attempt.evaluate_config(ret)
                population.append((ret, cost))

            while attempt.in_budget():
                population = sorted(population, key=lambda x: x[1])
                # keep best performing configs
                new_population = population[:elitism_size]

                for _ in range(self.population_size - elitism_size):
                    # choose parents from best performing configs
                    while True:
                        parent_one = random.choice(population[:reproduction_size])[0]
                        parent_two = random.choice(population[:reproduction_size])[0]

                        child = {}
                        for [k1, v1], [k2, v2] in zip(
                            parent_one.items(), parent_two.items()
                        ):
                            assert k1 == k2
                            # crossover and / or mutation
                            if random.random() < self.crossover_prob:
                                child[k1] = v1
                            else:
                                child[k1] = v2

                            if random.random() < self.mutation_prob:
                                if self.local_mutation:
                                    child[k1] = attempt.problem.params[
                                        k1
                                    ].choose_random(v1)
                                else:
                                    child[k1] = attempt.problem.params[
                                        k1
                                    ].choose_random()

                        if (
                            not self.check_constraints
                            or attempt.fulfills_all_constraints(child)
                        ):
                            break

                    if not attempt.in_budget():
                        return

                    cost = attempt.evaluate_config(child)
                    new_population.append((child, cost))

                population = new_population
        else:
            raise ValueError("Max. one schedule parameter is allowed")
