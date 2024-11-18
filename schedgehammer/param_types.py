import itertools
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, TypeVar

ParamValue = bool | float | int | str | list[int]
T = TypeVar("T", bound=ParamValue)


@dataclass
class Param(ABC, Generic[T]):
    @abstractmethod
    def choose_random(self) -> T:
        raise NotImplementedError

    @abstractmethod
    def get_value_range(self) -> list[T]:
        raise NotImplementedError


@dataclass
class SwitchParam(Param[bool]):
    def choose_random(self) -> bool:
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

    def get_value_range(self) -> list[int]:
        return list(range(self.min_val, self.max_val))


@dataclass
class ExpIntParam(Param[int]):
    base: int
    min_exp: int
    max_exp: int

    def choose_random(self) -> int:
        return self.base ** random.randint(self.min_exp, self.max_exp)

    def get_value_range(self) -> list[int]:
        # TODO: gucken, wo die parameter nicht als int angelegt werden
        return [self.base**i for i in range(int(self.min_exp), int(self.max_exp))]


OrdinalParamType = int | str


@dataclass
class OrdinalParam(Param[OrdinalParamType]):
    values: list[OrdinalParamType]

    def choose_random(self) -> OrdinalParamType:
        return random.choice(self.values)

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

    def choose_random(self) -> list[int]:
        tmp = self.values.copy()
        random.shuffle(tmp)
        return tmp

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
