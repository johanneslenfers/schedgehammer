from abc import ABC, abstractmethod
from dataclasses import dataclass
from itertools import permutations
from math import log2
from typing import Generic, TypeVar

T = TypeVar("T")


@dataclass
class ParamType(ABC, Generic[T]):
    name: str
    val_range: tuple[T]

    @abstractmethod
    def get_complexity_score(self) -> int:
        """
        The complexity score is roughly equal to the number of possible values the parameter can take.
        We further limit the complexity
        """
        raise NotImplementedError("get_complexity_score method is not implemented")


@dataclass
class SwitchParam(ParamType[bool]):
    def __init__(self, name: str):
        self.name = name
        self.val_range = [True, False]

    def get_complexity_score(self):
        return 2


@dataclass
class RealParam(ParamType[float]):
    pass

    def get_complexity_score(self):
        # We multiply the interval width with 3 and limit the complexity score to 30
        return max(self.val_range[1] - self.val_range[0] + 1, 10) * 3


@dataclass
class IntegerParam(ParamType[int]):
    pass

    def get_complexity_score(self):
        # Complexity score is Interval width but max. 10
        return max(self.val_range[1] - self.val_range[0] + 1, 10)


@dataclass
class OrdinalParam(ParamType[int]):
    def __post_init__(self):
        for val in self.val_range:
            # Check if all values are powers of 2
            # i.e. check if exactly one bit is set
            # (https://stackoverflow.com/questions/57025836/how-to-check-if-a-given-number-is-a-power-of-two)
            if val > 0 and not (val & (val - 1) == 0):
                raise ValueError("Ordinal val_range must be powers of 2")

    def get_complexity_score(self):
        return log2(self.val_range[1])


@dataclass
class CategoricalParam(ParamType[str]):
    def get_complexity_score(self):
        return len(self.val_range)


@dataclass
class PermutationParam(ParamType[int]):
    min_length: int
    max_length: int

    def get_complexity_score(self):
        return permutations(
            range(1, min(self.max_length, len(set(self.val_range))) + 1)
        )


TYPE_MAP = {
    "switch": SwitchParam,
    "real": RealParam,
    "integer": IntegerParam,
    "ordinal": OrdinalParam,
    "categorical": CategoricalParam,
    "permutation": PermutationParam,
}
