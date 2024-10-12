from dataclasses import dataclass
from typing import Callable
import random

from tuner.parameter import Config, SwitchParameter, RealParameter, IntegerParameter, OrdinalParameter, CategoricalParameter, PermutationParameter


@dataclass
class Tuner:
    config: Config
    budget: int = 100

    def get_dict(self) -> dict[str, bool | float | int | str | list[int]]:
        ret = {}

        for [name, param] in self.config.parameters.items():
            match param:
                case SwitchParameter():
                    ret[name] = True
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

        return ret

    def tune(self, cost_fn: Callable[[dict[str, any]], float]) -> float:
        min_dict = self.get_dict()
        min = cost_fn(min_dict)
        for _ in range(self.budget):
            cur_dict = self.get_dict()
            cur = cost_fn(cur_dict)

            min = cur if cur < min else min
            min_dict = cur_dict if cur < min else min_dict

        print(f"minimal cost: {min}")
        print(f"minimal dict: {min_dict}")
        return min
