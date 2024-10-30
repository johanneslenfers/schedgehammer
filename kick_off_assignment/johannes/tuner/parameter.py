from dataclasses import dataclass, field
from typing import Callable


@dataclass
class SwitchParameter:
    pass


@dataclass
class RealParameter:
    min: float
    max: float


@dataclass
class IntegerParameter:
    min: int
    max: int


@dataclass
class OrdinalParameter:
    vals: list[any]


@dataclass
class CategoricalParameter:
    vals: list[str]


@dataclass
class PermutationParameter:
    vals: list[int]


Parameter = (
    SwitchParameter
    | RealParameter
    | IntegerParameter
    | OrdinalParameter
    | CategoricalParameter
    | PermutationParameter
)


@dataclass
class SearchSpace:
    parameters: dict[str, Parameter]
    # all value parameters depend on key parameter
    dependencies: dict[str, list[str]] = field(default_factory=dict)
    constraints: dict[str, Callable[[any], bool]] = field(default_factory=dict)
