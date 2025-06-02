import random

from schedgehammer.tuner import Tuner, TuningAttempt

class GeneticTuner2(Tuner):
    population_size: int
    elitism_amount: int
    crossover_amount: int
    mutation_amount: int

    def __init__(
            self,
            population_size: int = 100,
            elitism_share: float = 0.1,
            crossover_share: float = 0.3,
            mutation_share: float = 0.4,
            randomgen_share: float = 0.2,
    ):
        self.population_size = population_size
        total = elitism_share + crossover_share + mutation_share + randomgen_share
        self.elitism_amount = round(elitism_share / total * self.population_size)
        self.crossover_amount = round(crossover_share / total * self.population_size)
        self.mutation_amount = round(mutation_share / total * self.population_size)

    def create_random(self, attempt: TuningAttempt):
        config = {}
        while True:
            for [name, param] in attempt.problem.params.items():
                config[name] = param.choose_random()

            if attempt.fulfills_all_constraints(config):
                return config

    def choose_weighted_random_from(self, population):
        max_value = max(population, key=lambda x: x[1])[1]
        weights = [max_value - cost for (config, cost) in population]
        return random.choices(population, weights)[0][0]  # <3

    def do_tuning(self, attempt: TuningAttempt):

        population = [self.create_random(attempt) for _ in range(self.population_size)]
        population_with_cost = [(config, attempt.evaluate_config(config)) for config in population]

        while True:
            prev_population = population_with_cost
            population_with_cost = prev_population[:self.elitism_amount]

            for i in range(self.population_size - self.elitism_amount):
                parent1 = self.choose_weighted_random_from(prev_population)
                parent2 = self.choose_weighted_random_from(prev_population)

                new_config = {}

                for name, param in attempt.problem.params.items():
                    if i < self.crossover_amount:
                        if param.can_crossover():
                            new_config[name] = param.crossover(parent1[name], parent2[name])
                            continue

                    if i < self.crossover_amount + self.mutation_amount:
                        if param.can_mutate():
                            select = parent1 if i >= self.crossover_amount else random.choice([parent1, parent2])
                            new_config[name] = param.mutate(select[name])
                            continue

                    new_config[name] = param.choose_random()

                if not attempt.in_budget():
                    return

                population_with_cost.append((new_config, attempt.evaluate_config(new_config)))

            population_with_cost.sort(key=lambda x: x[1])
