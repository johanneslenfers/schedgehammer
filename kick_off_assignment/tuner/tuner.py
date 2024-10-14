from dataclasses import dataclass, field
from typing import Callable
import random

from tuner.parameter import (
    SearchSpace,
    SwitchParameter,
    RealParameter,
    IntegerParameter,
    OrdinalParameter,
    CategoricalParameter,
    PermutationParameter,
)

ConfigDict = dict[str, bool | float | int | str | list[int]]


@dataclass
class Tuner:
    search_space: SearchSpace
    cost_fn: Callable[[dict[str, any]], float]
    results: list[tuple[float, ConfigDict]] = field(default_factory=list)
    budget: int = 100

    def random_sampling(self) -> ConfigDict:
        ret = {}

        for [name, param] in self.search_space.parameters.items():
            match param:
                case SwitchParameter():
                    ret[name] = True if random.random() < 0.5 else False
                case RealParameter(min, max):
                    ret[name] = min + random.random() * (max - min)
                case IntegerParameter(min, max):
                    ret[name] = random.randint(min, max)
                case OrdinalParameter(vals):
                    ret[name] = vals[random.randint(0, len(vals) - 1)]
                case CategoricalParameter(vals):
                    ret[name] = vals[random.randint(0, len(vals) - 1)]
                case PermutationParameter(vals):
                    tmp = vals.copy()
                    random.shuffle(tmp)
                    ret[name] = tmp

        cost = self.cost_fn(ret)
        self.results.append((cost, ret))

    def tune(self):
        for _ in range(self.budget):
            self.random_sampling()

        min, min_dict = sorted(self.results, key=lambda x: x[0])[0]

        print(f"minimal cost: {min}")
        print(f"minimal dict: {min_dict}")
