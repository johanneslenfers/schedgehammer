from dataclasses import dataclass
from typing import Callable

from tuner.parameter import *

@dataclass
class Tuner:
    config: Config
    budget: int = 50

    def get_dict(self) -> dict[str, bool | float | int | str | list[int]]:
        ret = {}

        for [name, param] in self.config.parameters.items():
            match param:
                case SwitchParameter(): ret[name] = True
                case RealParameter(min, max): ret[name] = min + random.random() * (max - min)
                case IntegerParameter(min, max): ret[name] = random.randint(min, max)
                case OrdinalParameter(vals): ret[name] = vals[random.randint(0, len(vals) - 1)]
                case CategoricalParameter(vals): ret[name] = vals[random.randint(0, len(vals) - 1)]
                case PermutationParameter(vals): tmp = vals.copy() ;shuffle(tmp) ;ret[name] = tmp

        return ret

    def tune(self, cost_fn: Callable[[dict[str, any]], float]) -> float:
        min = cost_fn(self.get_dict())
        for _ in range(self.budget):
            cur = cost_fn(self.get_dict())

            min = cur if cur < min else min

        print(f"minimal cost: {min}")
        return min

            
