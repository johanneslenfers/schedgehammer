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

    def do_tuning(self, attempt: TuningAttempt):
        elitism_size = int(self.population_size * self.elitism_share)
        reproduction_size = int(self.population_size * self.reproduction_share)

        # initial population
        population = []
        for _ in range(self.population_size):
            ret = {}
            for [name, param] in attempt.problem.params.items():
                ret[name] = param.choose_random()

            cost = attempt.evaluate_config(ret)
            population.append((ret, cost))

        while attempt.in_budget():
            population = sorted(population, key=lambda x: x[1])
            # keep best performing configs
            new_population = population[:elitism_size]

            for _ in range(self.population_size - elitism_size):
                # choose parents from best performing configs
                parent_one = random.choice(population[:reproduction_size])[0]
                parent_two = random.choice(population[:reproduction_size])[0]

                child = {}
                for [k1, v1], [k2, v2] in zip(parent_one.items(), parent_two.items()):
                    assert k1 == k2
                    # crossover and / or mutation
                    if random.random() < self.crossover_prob:
                        child[k1] = v1
                    else:
                        child[k1] = v2

                    if random.random() < self.mutation_prob:
                        child[k1] = attempt.problem.params[k1].choose_random()

                cost = attempt.evaluate_config(child)
                new_population.append((child, cost))

            population = new_population
            attempt.log_state()
