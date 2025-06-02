import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, Optional, TypeVar
import itertools

ParamValue = bool | float | int | str | list[int]
T = TypeVar("T", bound=ParamValue)


def clamp(val: T, min_val: T, max_val: T) -> T:
    return max(min_val, min(val, max_val))


@dataclass
class Param(ABC, Generic[T]):

    @abstractmethod
    def choose_random(self) -> T:
        raise NotImplementedError

    def can_mutate(self) -> bool:
        return False

    def mutate(self, value: T) -> T:
        raise NotImplementedError

    def can_crossover(self) -> bool:
        return False

    def crossover(self, value1: T, value2: T) -> T:
        raise NotImplementedError

    def get_value_range(self) -> list[T]:
        raise NotImplementedError

    def translate_for_evaluation(self, value: T) -> T:
        return value


@dataclass
class SwitchParam(Param[bool]):
    def choose_random(self, current_value: Optional[bool] = None) -> bool:
        if random.random() < 0.5:
            return True
        else:
            return False

    def get_value_range(self) -> list[bool]:
        return [True, False]


@dataclass
class RealParam(Param[float]):
    min_val: float
    max_val: float
    range_precision: float = 1e-3

    def choose_random(self) -> float:
        return random.uniform(self.min_val, self.max_val)

    def can_mutate(self) -> bool:
        return True

    def mutate(self, value: T) -> T:
        stddev = (self.max_val - self.min_val) / 4
        return clamp(
            random.normalvariate(value, stddev), self.min_val, self.max_val
        )

    def get_value_range(self) -> list[float]:
        return [
            self.min_val + n * self.range_precision
            for n in range(
                int((self.max_val - self.min_val) * (1 / self.range_precision))
            )
        ]


@dataclass
class IntegerParam(Param[int]):
    min_val: int
    max_val: int

    def choose_random(self) -> int:
        return random.randint(self.min_val, self.max_val)

    def can_mutate(self) -> bool:
        return True

    def mutate(self, value: T) -> T:
        stddev = (self.max_val - self.min_val) / 4
        return clamp(
            round(random.normalvariate(value, stddev)), self.min_val, self.max_val
        )

    def get_value_range(self) -> list[int]:
        return list(range(self.min_val, self.max_val))


@dataclass
class ExpIntParam(Param[int]):
    base: int
    min_exp: int
    max_exp: int

    def choose_random(self) -> int:
        return self.base ** random.randint(self.min_exp, self.max_exp)

    def can_mutate(self) -> bool:
        return True

    def mutate(self, value: T) -> T:
        stddev = (self.max_exp - self.min_exp) / 4
        exp = clamp(
            round(random.normalvariate(value, stddev)),
            self.min_exp,
            self.max_exp,
        )
        return self.base ** exp

    def get_value_range(self) -> list[int]:
        # TODO: gucken, wo die parameter nicht als int angelegt werden
        return [self.base**i for i in range(int(self.min_exp), int(self.max_exp))]


OrdinalParamType = int | str


@dataclass
class OrdinalParam(Param[OrdinalParamType]):
    values: list[OrdinalParamType]

    def choose_random(self) -> OrdinalParamType:
        return random.choice(self.values)

    def can_mutate(self) -> bool:
        return True

    def mutate(self, value: T) -> T:
        stddev = len(self.values) / 4
        current_idx = self.values.index(value)
        idx = clamp(
            int(random.normalvariate(current_idx, stddev)), 0, len(self.values) - 1
        )
        return self.values[idx]

    def get_value_range(self) -> list[OrdinalParamType]:
        return self.values.copy()


@dataclass
class CategoricalParam(Param[str]):
    values: list[str]

    def choose_random(self) -> str:
        return random.choice(self.values)

    def get_value_range(self) -> list[str]:
        return self.values.copy()


@dataclass
class PermutationParam(Param[list[int]]):
    values: list[int]

    def choose_random(self, current_value: Optional[list[int]] = None) -> list[int]:
        tmp = self.values.copy()
        random.shuffle(tmp)
        return tmp

    def can_mutate(self) -> bool:
        return True

    def mutate(self, value: T) -> T:
        stddev = len(self.values) / 4
        swaps = clamp(
            int(random.normalvariate(len(self.values) / 2, stddev)),
            0,
            len(self.values) - 1,
        )

        for _ in range(swaps):
            idx1 = random.randint(0, len(self.values) - 1)
            idx2 = random.randint(0, len(self.values) - 1)

            value[idx1], value[idx2] = value[idx2], value[idx1]

        return value

    def get_value_range(self) -> list[list[int]]:
        return [list(p) for p in itertools.permutations(self.values)]


TYPE_MAP = {
    "switch": SwitchParam,
    "real": RealParam,
    "integer": IntegerParam,
    "ordinal": OrdinalParam,
    "categorical": CategoricalParam,
    "permutation": PermutationParam,
}
