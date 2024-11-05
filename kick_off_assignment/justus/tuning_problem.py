import random
from abc import ABCMeta, abstractmethod
from typing import List, Callable

TuningParameterValue = bool | float | int | str | list[int]
TuningConfig = dict[str, TuningParameterValue]


class TuningParameter(metaclass=ABCMeta):
    @abstractmethod
    def random_value(self) -> TuningParameterValue:
        pass

    @abstractmethod
    def mutate_value(self, value: TuningParameterValue) -> TuningParameterValue:
        pass

    @abstractmethod
    def cross_values(self, value1: TuningParameterValue, value2: TuningParameterValue) -> TuningParameterValue:
        pass


class BooleanParameter(TuningParameter):

    def random_value(self) -> bool:
        return random.choice([True, False])

    def mutate_value(self, value: bool) -> bool:
        return not value

    def cross_values(self, value1: bool, value2: bool) -> bool:
        return random.choice([value1, value2])


class IntegerParameter(TuningParameter):
    min_value: int
    max_value: int

    def __init__(self, min_value: int, max_value: int):
        self.min_value = min_value
        self.max_value = max_value
        super().__init__()

    def random_value(self) -> int:
        return random.randint(self.min_value, self.max_value)

    def mutate_value(self, value: int) -> int:
        pass

    def cross_values(self, value1: int, value2: int) -> int:
        pass


class RealParameter(TuningParameter):
    min_value: float
    max_value: float

    def __init__(self, min_value: float, max_value: float):
        self.min_value = min_value
        self.max_value = max_value
        super().__init__()
    
    def random_value(self) -> float:
        return random.uniform(self.min_value, self.max_value)

    def mutate_value(self, value: float) -> float:
        pass

    def cross_values(self, value1: float, value2: float) -> float:
        pass


class OrdinalParameter(TuningParameter):
    values: List[int]

    def __init__(self, values: List[int]):
        self.values = values
        super().__init__()
    
    def random_value(self) -> int:
        return random.choice(self.values)

    def mutate_value(self, value: int) -> int:
        pass

    def cross_values(self, value1: int, value2: int) -> int:
        pass


class CategoricalParameter(TuningParameter):
    values: List[str]

    def __init__(self, values: List[str]):
        self.values = values
        super().__init__()
    
    def random_value(self) -> str:
        return random.choice(self.values)

    def mutate_value(self, value: str) -> str:
        pass

    def cross_values(self, value1: str, value2: str) -> str:
        pass


class PermutationParameter(TuningParameter):
    size: int

    def __init__(self, size: int):
        self.size = size
        super().__init__()
    
    def random_value(self) -> List[int]:
        l = list(range(1, self.size + 1))
        random.shuffle(l)
        return l

    def mutate_value(self, value: List[int]) -> List[int]:
        pass

    def cross_values(self, value1: List[int], value2: List[int]) -> List[int]:
        pass


class TuningProblem(metaclass=ABCMeta):
    parameters: dict[str, TuningParameter]
    cost: Callable[[dict[str, TuningParameterValue]], float]

    def __init__(self, parameters: dict[str, TuningParameter], cost: Callable[[dict[str, TuningParameterValue]], float]):
        self.parameters = parameters
        self.cost = cost
