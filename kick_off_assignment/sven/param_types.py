from abc import ABC, abstractmethod
from dataclasses import dataclass
from itertools import permutations, product
from math import ceil, log2, prod
from typing import Generic, Iterator, TypeVar

import numpy as np

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

    @abstractmethod
    def get_n_samples(self, n: int) -> Iterator[T]:
        """
        Returns n samples from the parameter's value range.
        """
        raise NotImplementedError("get_n_samples method is not implemented")


@dataclass
class SwitchParam(ParamType[bool]):
    def __init__(self, name: str):
        self.name = name
        self.val_range = [True, False]

    def get_complexity_score(self):
        return 2

    def get_n_samples(self, n):
        for _ in range(int(n // 2)):
            yield False
        for _ in range(ceil(n / 2)):
            yield True


@dataclass
class RealParam(ParamType[float]):
    pass

    def get_complexity_score(self):
        # We multiply the interval width with 'magic' factor 3 and limit the complexity score to 30
        return max(self.val_range[1] - self.val_range[0] + 1, 10) * 3

    def get_n_samples(self, n):
        for val in np.linspace(self.val_range[0], self.val_range[1], n):
            yield val


@dataclass
class IntegerParam(ParamType[int]):
    pass

    def get_complexity_score(self):
        # Complexity score is Interval width but max. 10
        return max(self.val_range[1] - self.val_range[0] + 1, 10)

    def get_n_samples(self, n):
        for val in np.linspace(self.val_range[0], self.val_range[1], n, dtype=int):
            yield val


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

    def get_n_samples(self, n):
        ord_vals_in_range = [
            2**i
            for i in range(
                int(log2(self.val_range[0])), int(log2(self.val_range[1])) + 1
            )
        ]
        idxs = np.linspace(0, len(ord_vals_in_range) - 1, n, dtype=int)
        for idx in idxs:
            yield ord_vals_in_range[idx]


@dataclass
class CategoricalParam(ParamType[str]):
    def get_complexity_score(self):
        return len(self.val_range)

    def get_n_samples(self, n):
        for val in np.linspace(0, len(self.val_range) - 1, n, dtype=int):
            yield self.val_range[val]


@dataclass
class PermutationParam(ParamType[list[int]]):
    def get_bounds(self) -> tuple[list[int], list[int]]:
        return self.val_range[0], self.val_range[1]

    def get_complexity_score(self):
        lower_bounds, upper_bounds = self.get_bounds()
        return max(
            prod(up - lower for up, lower in zip(upper_bounds, lower_bounds)), 70
        )

    def get_n_samples(self, n):
        lower_bounds, upper_bounds = self.get_bounds()
        linspaces = [
            np.linspace(lower, upper, n, dtype=int)
            for lower, upper in zip(lower_bounds, upper_bounds)
        ]
        for idx in range(n):
            sample = [linspace[idx] for linspace in linspaces]
            yield sample


TYPE_MAP = {
    "switch": SwitchParam,
    "real": RealParam,
    "integer": IntegerParam,
    "ordinal": OrdinalParam,
    "categorical": CategoricalParam,
    "permutation": PermutationParam,
}
