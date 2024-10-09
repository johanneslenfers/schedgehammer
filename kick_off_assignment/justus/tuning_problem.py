from abc import ABCMeta, abstractmethod
from typing import List

class TuningParameter(metaclass=ABCMeta):
    pass

class BooleanParameter(TuningParameter):
    def __init__(self):
        pass


class IntegerParameter(TuningParameter):
    min_value: int
    max_value: int

    def __init__(self, min_value: int, max_value: int):
        self.min_value = min_value
        self.max_value = max_value
        super().__init__()


class RealParameter(TuningParameter):
    min_value: float
    max_value: float

    def __init__(self, min_value: float, max_value: float):
        self.min_value = min_value
        self.max_value = max_value
        super().__init__()


class OrdinalParameter(TuningParameter):
    values: List[int]

    def __init__(self, values: List[int]):
        self.values = values
        super().__init__()


class CategoricalParameter(TuningParameter):
    values: List[str]

    def __init__(self, values: List[str]):
        self.values = values
        super().__init__()


class PermutationParameter(TuningParameter):
    size: int

    def __init__(self, size: int):
        self.size = size
        super().__init__()

TuningParameterValue = bool | float | int | str | list[int]

class TuningProblem(metaclass=ABCMeta):
    @abstractmethod
    def config(self) -> dict[str, TuningParameter]:
        pass

    @abstractmethod
    def cost(self, configuration: dict[str, TuningParameterValue]) -> float:
        pass