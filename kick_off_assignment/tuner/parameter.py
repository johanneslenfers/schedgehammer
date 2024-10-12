from dataclasses import dataclass
from random import shuffle
import random


@dataclass
class Parameter:
    pass


@dataclass
class SwitchParameter(Parameter):
    pass


@dataclass
class RealParameter(Parameter):
    min: float
    max: float


@dataclass
class IntegerParameter(Parameter):
    min: int
    max: int


@dataclass
class OrdinalParameter(Parameter):
    vals: list[any]


@dataclass
class CategoricalParameter(Parameter):
    vals: list[str]


@dataclass
class PermutationParameter(Parameter):
    vals: list[int]


@dataclass
class Config:
    parameters: dict[str, Parameter]
