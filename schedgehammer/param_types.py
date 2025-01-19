import random
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, TypeVar, Optional
from numbers import Number
import itertools

ParamValue = bool | float | int | str | list[int]
T = TypeVar("T", bound=ParamValue)


def clamp(val: Number, min_val: Number, max_val: Number) -> Number:
    return max(min_val, min(val, max_val))


@dataclass
class Param(ABC, Generic[T]):
    @abstractmethod
    def choose_random(self, current_value: Optional[T] = None) -> T:
        raise NotImplementedError

    @abstractmethod
    def get_value_range(self) -> list[T]:
        raise NotImplementedError


@dataclass
class SwitchParam(Param[bool]):
    def choose_random(self, current_value: Optional[bool] = None) -> bool:
        if random.random() < 0.5:
            return True
        else:
            return False

    def get_value_range(self) -> bool:
        return np.array([True, False])


@dataclass
class RealParam(Param[float]):
    min_val: float
    max_val: float
    range_precision: float = 1e-3

    def choose_random(self, current_value: Optional[float] = None) -> float:
        if current_value is None:
            return random.uniform(self.min_val, self.max_val)
        else:
            stddev = (self.max_val - self.min_val) / 4
            return clamp(
                random.normalvariate(current_value, stddev), self.min_val, self.max_val
            )

    def get_value_range(self) -> list[float]:
        return np.array(
            [
                self.min_val + n * self.range_precision
                for n in range(
                    int((self.max_val - self.min_val) * (1 / self.range_precision))
                )
            ]
        )


@dataclass
class IntegerParam(Param[int]):
    min_val: int
    max_val: int

    def choose_random(self, current_value: Optional[int] = None) -> int:
        if current_value is None:
            return random.randint(self.min_val, self.max_val)
        else:
            stddev = (self.max_val - self.min_val) / 4
            return clamp(
                int(random.normalvariate(current_value, stddev)),
                self.min_val,
                self.max_val,
            )

    def get_value_range(self) -> list[int]:
        return np.array(list(range(self.min_val, self.max_val)))


@dataclass
class ExpIntParam(Param[int]):
    base: int
    min_exp: int
    max_exp: int

    def choose_random(self, current_value: Optional[int] = None) -> int:
        if current_value is None:
            return self.base ** random.randint(self.min_exp, self.max_exp)
        else:
            stddev = (self.max_exp - self.min_exp) / 4
            exp = clamp(
                int(random.normalvariate(current_value, stddev)),
                self.min_exp,
                self.max_exp,
            )
            return self.base**exp

    def get_value_range(self) -> int:
        # TODO: gucken, wo die parameter nicht als int angelegt werden
        return np.array(
            [self.base**i for i in range(int(self.min_exp), int(self.max_exp))]
        )


OrdinalParamType = int | str


@dataclass
class OrdinalParam(Param[OrdinalParamType]):
    values: list[OrdinalParamType]

    def choose_random(
        self, current_value: Optional[OrdinalParamType] = None
    ) -> OrdinalParamType:
        if current_value is None:
            return random.choice(self.values)
        else:
            stddev = len(self.values) / 4
            current_idx = self.values.index(current_value)
            idx = clamp(
                int(random.normalvariate(current_idx, stddev)), 0, len(self.values) - 1
            )
            return self.values[idx]

    def get_value_range(self) -> list[OrdinalParamType]:
        return np.array(self.values.copy())


@dataclass
class CategoricalParam(Param[str]):
    values: list[str]

    def choose_random(self, _: Optional[str] = None) -> str:
        return random.choice(self.values)

    def get_value_range(self) -> list[str]:
        return np.array(self.values.copy())


@dataclass
class PermutationParam(Param[list[int]]):
    values: list[int]

    def choose_random(self, current_value: Optional[list[int]] = None) -> list[int]:
        if current_value is None:
            tmp = self.values.copy()
            random.shuffle(tmp)
            return tmp
        else:
            stddev = len(self.values) / 4
            swaps = clamp(
                int(random.normalvariate(len(self.values) / 2, stddev)),
                0,
                len(self.values) - 1,
            )

            for _ in range(swaps):
                idx1 = random.randint(0, len(self.values) - 1)
                idx2 = random.randint(0, len(self.values) - 1)

                current_value[idx1], current_value[idx2] = (
                    current_value[idx2],
                    current_value[idx1],
                )

            return current_value

    def get_value_range(self) -> list[int]:
        return np.array(
            [tuple(p) for p in itertools.permutations(self.values)],
            dtype=[(str(i), np.int32) for i in range(len(self.values))],
        )


TYPE_MAP = {
    "switch": SwitchParam,
    "real": RealParam,
    "integer": IntegerParam,
    "ordinal": OrdinalParam,
    "categorical": CategoricalParam,
    "permutation": PermutationParam,
}
