import math
import random
from dataclasses import dataclass, field
from typing import Callable

from tuner.parameter import (
    CategoricalParameter,
    IntegerParameter,
    OrdinalParameter,
    Parameter,
    PermutationParameter,
    RealParameter,
    SearchSpace,
    SwitchParameter,
)

ParamValue = str, bool | float | int | str | list[int]
ConfigDict = dict[ParamValue]


def random_param_value(param: Parameter) -> ParamValue:
    match param:
        case SwitchParameter():
            return random.random() < 0.5
        case RealParameter(min, max):
            return min + random.random() * (max - min)
        case IntegerParameter(min, max):
            return random.randint(min, max)
        case OrdinalParameter(vals):
            return vals[random.randint(0, len(vals) - 1)]
        case CategoricalParameter(vals):
            return vals[random.randint(0, len(vals) - 1)]
        case PermutationParameter(vals):
            tmp = vals.copy()
            random.shuffle(tmp)
            return tmp


@dataclass
class Tuner:
    search_space: SearchSpace
    cost_fn: Callable[[ConfigDict], float]
    results: list[tuple[float, ConfigDict, int]] = field(default_factory=list)
    budget: int = 10000

    def random_sampling(self) -> tuple[float, ConfigDict]:
        for _ in range(self.budget):
            ret = {}
            for [name, param] in self.search_space.parameters.items():
                ret[name] = random_param_value(param)

            cost = self.cost_fn(ret)
            self.results.append((cost, ret))

        return sorted(self.results, key=lambda x: x[0])[0]

    def genetic_algo(
        self,
        generations: int = 100,
        elitism_share: float = 0.1,
        reproduction_share: float = 0.3,
        crossover_prob: float = 0.5,
        mutation_prob: float = 0.1,
    ) -> tuple[float, ConfigDict, int]:
        population_size = math.ceil(self.budget / generations)
        elitism_size = math.ceil(population_size * elitism_share)
        reproduction_size = math.ceil(population_size * reproduction_share)
        num_evaluations = 0

        # initial population
        population = []
        for _ in range(population_size):
            ret = {}
            for [name, param] in self.search_space.parameters.items():
                ret[name] = random_param_value(param)

            num_evaluations += 1
            cost = self.cost_fn(ret)
            population.append((cost, ret, num_evaluations))

        self.results.extend(population)

        for i in range(generations - 1):
            population = sorted(population, key=lambda x: x[0])
            # keep best performing configs
            new_population = population[:elitism_size]

            for _ in range(population_size - elitism_size):
                # choose parents from best performing configs
                parent_one = random.choice(population[:reproduction_size])[1]
                parent_two = random.choice(population[:reproduction_size])[1]

                child = {}
                for [k1, v1], [k2, v2] in zip(parent_one.items(), parent_two.items()):
                    assert k1 == k2
                    # crossover and / or mutation
                    if random.random() < crossover_prob:
                        child[k1] = v1
                    else:
                        child[k1] = v2

                    if random.random() < mutation_prob:
                        child[k1] = random_param_value(self.search_space.parameters[k1])

                num_evaluations += 1
                cost = self.cost_fn(child)
                new_population.append((cost, child, num_evaluations))

            population = new_population
            self.results.extend(population)

        return population[0]
