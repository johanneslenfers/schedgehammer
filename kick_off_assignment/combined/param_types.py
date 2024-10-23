import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, TypeVar

ParamValueTypes = bool | float | int | str | list[int]
T = TypeVar("T", bound=ParamValueTypes)


@dataclass
class Param(ABC, Generic[T]):
    name: str
    val_range: tuple[T]
    val: T = None

    @abstractmethod
    def choose_random(self):
        raise NotImplementedError

    @abstractmethod
    def get_neighbours(self) -> list[T]:
        raise NotImplementedError


@dataclass
class SwitchParam(Param[bool]):
    def choose_random(self):
        self.val = self.val_range[random.randint(0, 1)]

    def get_neighbours(self):
        return [not self.val]


@dataclass
class RealParam(Param[float]):
    def choose_random(self):
        self.val = random.uniform(self.val_range[0], self.val_range[1])

    def get_neighbours(self):
        epsilon = (self.val_range[1] - self.val_range[0]) / 1000
        neighbours = []
        if self.val - epsilon >= self.val_range[0]:
            neighbours.append(self.val - epsilon)
        if self.val + epsilon <= self.val_range[1]:
            neighbours.append(self.val + epsilon)
        return neighbours


@dataclass
class IntegerParam(Param[int]):
    def choose_random(self):
        self.val = random.randint(self.val_range[0], self.val_range[1])

    def get_neighbours(self):
        neighbours = []
        if self.val - 1 >= self.val_range[0]:
            neighbours.append(self.val - 1)
        if self.val + 1 <= self.val_range[1]:
            neighbours.append(self.val + 1)
        return neighbours


@dataclass
class OrdinalParam(Param[int | str]):
    def choose_random(self):
        self.val = self.val_range[random.randint(0, len(self.val_range) - 1)]

    def get_neighbours(self):
        i = self.val_range.index(self.val)
        neighbours = []
        if i > 0:
            neighbours.append(self.val_range[i - 1])
        if i < len(self.val_range) - 1:
            neighbours.append(self.val_range[i + 1])
        return neighbours


@dataclass
class CategoricalParam(Param[int | str]):
    def choose_random(self):
        self.val = self.val_range[random.randint(0, len(self.val_range) - 1)]

    def get_neighbours(self):
        return [val for val in self.val_range if val != self.val]


@dataclass
class PermutationParam(Param[list[int]]):
    def choose_random(self):
        lower_bound, upper_bound = self.value_range
        self.val = [
            random.randint(lower_bound[i], upper_bound[i])
            for i in range(len(lower_bound))
        ]

    def get_neighbours(self):
        neighbours = []
        for i in range(len(self.val)):
            if self.val[i] - 1 >= self.val_range[0][i]:
                new_val = self.val.copy()
                new_val[i] -= 1
                neighbours.append(new_val)
            if self.val[i] + 1 <= self.val_range[1][i]:
                new_val = self.val.copy()
                new_val[i] += 1
                neighbours.append(new_val)
        return neighbours


TYPE_MAP = {
    "switch": SwitchParam,
    "real": RealParam,
    "integer": IntegerParam,
    "ordinal": OrdinalParam,
    "categorical": CategoricalParam,
    "permutation": PermutationParam,
}
