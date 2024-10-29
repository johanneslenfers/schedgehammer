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


@dataclass
class SwitchParam(Param[bool]):
    def choose_random(self):
        if random.random() < 0.5:
            return True
        else:
            return False


@dataclass
class RealParam(Param[float]):
    min_val: float
    max_val: float

    def choose_random(self) -> float:
        return random.uniform(self.min_val, self.max_val)


@dataclass
class IntegerParam(Param[int]):
    min_val: int
    max_val: int

    def choose_random(self) -> int:
        return random.randint(self.min_val, self.max_val)


OrdinalParamType = int | str


@dataclass
class OrdinalParam(Param[OrdinalParamType]):
    values: list[OrdinalParamType]

    def choose_random(self) -> OrdinalParamType:
        return random.choice(self.values)


@dataclass
class CategoricalParam(Param[str]):
    values: list[str]

    def choose_random(self) -> str:
        return random.choice(self.values)


@dataclass
class PermutationParam(Param[list[int]]):
    values: list[int]

    def choose_random(self) -> list[int]:
        tmp = self.values.copy()
        random.shuffle(tmp)
        return tmp


TYPE_MAP = {
    "switch": SwitchParam,
    "real": RealParam,
    "integer": IntegerParam,
    "ordinal": OrdinalParam,
    "categorical": CategoricalParam,
    "permutation": PermutationParam,
}
