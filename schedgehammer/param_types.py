import math
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from http.cookiejar import domain_match
from typing import Generic, TypeVar, Optional
from numbers import Number
import itertools

ParamValue = bool | float | int | str | list[int]
T = TypeVar("T", bound=ParamValue)


def clamp(val: Number, min_val: Number, max_val: Number) -> Number:
    return max(min_val, min(val, max_val))

def binary_search(array, v):
    low = 0
    high = len(array) - 1
    while low <= high:
        mid = (low + high) // 2
        if array[mid] == v:
            return mid
        elif array[mid] < v:
            low = mid + 1
        else:
            high = mid - 1
    return low



@dataclass
class Param(ABC, Generic[T]):
    @abstractmethod
    def choose_random(self, current_value: Optional[T] = None) -> T:
        raise NotImplementedError

    def has_manageable_discrete_range(self) -> bool:
        return False

    @abstractmethod
    def get_value_range(self) -> list[T]:
        """
        Has to be implemented if has_manageable_discrete_range returns True.
        """
        raise NotImplementedError

    def choose_random_around_in(self, around_value: T, domain: list[T]):
        current_index = binary_search(domain, around_value)
        l = len(domain)

        idx = clamp(
            round(random.normalvariate(current_index, l / 4)), 0, l - 1
        )
        return domain[idx]


@dataclass
class BooleanParam(Param[bool]):
    def choose_random(self, current_value: Optional[bool] = None) -> bool:
        if random.random() < 0.5:
            return True
        else:
            return False

    def has_manageable_discrete_range(self) -> bool:
        return True

    def get_value_range(self) -> list[bool]:
        return [True, False]


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

    def has_manageable_discrete_range(self) -> bool:
        return True

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

    def choose_random(self, current_value: Optional[int] = None) -> int:
        if current_value is None:
            return random.randint(self.min_val, self.max_val)
        else:
            stddev = (self.max_val - self.min_val) / 4
            return clamp(
                round(random.normalvariate(current_value, stddev)),
                self.min_val,
                self.max_val,
            )

    def has_manageable_discrete_range(self) -> bool:
        return True

    def get_value_range(self) -> list[int]:
        return list(range(self.min_val, self.max_val))


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
                round(random.normalvariate(math.log(current_value, self.base), stddev)),
                self.min_exp,
                self.max_exp,
            )
            return self.base**exp

    def has_manageable_discrete_range(self) -> bool:
        return True

    def get_value_range(self) -> int:
        # TODO: gucken, wo die parameter nicht als int angelegt werden
        return [self.base**i for i in range(int(self.min_exp), int(self.max_exp))]


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
                round(random.normalvariate(current_idx, stddev)), 0, len(self.values) - 1
            )
            return self.values[idx]

    def has_manageable_discrete_range(self) -> bool:
        return True

    def get_value_range(self) -> list[OrdinalParamType]:
        return self.values.copy()


@dataclass
class CategoricalParam(Param[str]):
    values: list[str]

    def choose_random(self, _: Optional[str] = None) -> str:
        return random.choice(self.values)

    def has_manageable_discrete_range(self) -> bool:
        return True

    def get_value_range(self) -> list[str]:
        return self.values.copy()


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
                round(random.normalvariate(len(self.values) / 2, stddev)),
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

    def has_manageable_discrete_range(self) -> bool:
        # TODO: maybe only for small len(self.values)?
        return True

    def get_value_range(self) -> list[list[int]]:
        return [list(p) for p in itertools.permutations(self.values)]
