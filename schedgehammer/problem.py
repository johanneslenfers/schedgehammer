from typing import Callable

from schedgehammer.constraint import Constraint
from schedgehammer.param_types import Param, ParamValue


class Problem:
    name: str
    params: dict[str, Param]
    cost_function: Callable[[dict[str, ParamValue]], float]
    constraints: list[Constraint]

    def __init__(self,
                 name: str,
                 params: dict[str, Param],
                 cost_function: Callable[[dict[str, ParamValue]], float],
                 constraints: list[tuple[str, list[str]]] = []):
        self.name = name
        self.params = params
        self.cost_function = cost_function
        self.constraints = [Constraint(c, d) for (c, d) in constraints]
