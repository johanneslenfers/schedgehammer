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
    def distance(self, value1: TuningParameterValue, value2: TuningParameterValue) -> float:
        pass


class BooleanParameter(TuningParameter):

    def random_value(self) -> bool:
        return random.choice([True, False])

    def distance(self, value1: bool, value2: bool) -> float:
        if value1 != value2:
            return 1
        else:
            return 0


class IntegerParameter(TuningParameter):
    min_value: int
    max_value: int

    def __init__(self, min_value: int, max_value: int):
        self.min_value = min_value
        self.max_value = max_value
        super().__init__()

    def random_value(self) -> int:
        return random.randint(self.min_value, self.max_value)

    def distance(self, value1: int, value2: int) -> float:
        return abs(value1 - value2)


class RealParameter(TuningParameter):
    min_value: float
    max_value: float

    def __init__(self, min_value: float, max_value: float):
        self.min_value = min_value
        self.max_value = max_value
        super().__init__()
    
    def random_value(self) -> float:
        return random.uniform(self.min_value, self.max_value)

    def distance(self, value1: float, value2: float) -> float:
        return abs(value1 - value2)


class OrdinalParameter(TuningParameter):
    values: List[int]

    def __init__(self, values: List[int]):
        self.values = values
        super().__init__()
    
    def random_value(self) -> int:
        return random.choice(self.values)

    def distance(self, value1: int, value2: int) -> float:
        return abs(self.values.index(value1) - self.values.index(value2))


class CategoricalParameter(TuningParameter):
    values: List[str]

    def __init__(self, values: List[str]):
        self.values = values
        super().__init__()
    
    def random_value(self) -> str:
        return random.choice(self.values)

    def distance(self, value1: int, value2: int) -> float:
        if value1 != value2:
            return 1
        else:
            return 0


class PermutationParameter(TuningParameter):
    size: int

    def __init__(self, size: int):
        self.size = size
        super().__init__()
    
    def random_value(self) -> List[int]:
        l = list(range(1, self.size + 1))
        random.shuffle(l)
        return l

    def distance(self, value1: List[int], value2: List[int]) -> float:
        # Spearman
        return sum([(value1.index(i) - value2.index(i)) ** 2 for i in range(1, self.size + 1)])

class TuningProblem(metaclass=ABCMeta):
    parameters: dict[str, TuningParameter]
    cost: Callable[[dict[str, TuningParameterValue]], float]

    def __init__(self, parameters: dict[str, TuningParameter], cost: Callable[[dict[str, TuningParameterValue]], float]):
        self.parameters = parameters
        self.cost = cost
