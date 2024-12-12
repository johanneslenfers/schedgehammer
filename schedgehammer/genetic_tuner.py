import random
from dataclasses import dataclass

from schedgehammer.tuner import Tuner, TuningAttempt


@dataclass
class GeneticTuner(Tuner):
    population_size: int = 100
    elitism_share: float = 0.1
    reproduction_share: float = 0.3
    crossover_prob: float = 0.5
    mutation_prob: float = 0.1
    local_mutation: bool = False

    def do_tuning(self, attempt: TuningAttempt):
        elitism_size = int(self.population_size * self.elitism_share)
        reproduction_size = int(self.population_size * self.reproduction_share)

        # initial population
        initial = [next(attempt.solver.solve()) for _ in range(self.population_size)]
        population = [(ret, attempt.evaluate_config(ret)) for ret in initial]

        while attempt.in_budget():
            population = sorted(population, key=lambda x: x[1])
            # keep best performing configs
            new_population = population[:elitism_size]

            for _ in range(self.population_size - elitism_size):
                # choose parents from best performing configs
                parent_one = random.choice(population[:reproduction_size])[0]
                parent_two = random.choice(population[:reproduction_size])[0]

                for [k, v1], [_, v2] in zip(parent_one.items(), parent_two.items()):
                    # crossover and / or mutation
                    if random.random() < self.crossover_prob:
                        attempt.solver.decision_queue.append((k, v1))
                    else:
                        attempt.solver.decision_queue.append((k, v2))

                    if random.random() < self.mutation_prob:
                        if self.local_mutation:
                            attempt.solver.decision_queue.append(
                                (k, attempt.problem.params[k].choose_random(v1))
                            )
                        else:
                            attempt.solver.decision_queue.append(
                                (k, attempt.problem.params[k].choose_random())
                            )

                child = next(attempt.solver.solve())

                if not attempt.in_budget():
                    return

                cost = attempt.evaluate_config(child)
                new_population.append((child, cost))

            population = new_population
