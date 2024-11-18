from typing import Callable

from schedgehammer.constraint import ConstraintExpression
from schedgehammer.param_types import Param, ParamValue


class Problem:
    name: str
    params: dict[str, Param]
    cost_function: Callable[[dict[str, ParamValue]], float]
    constraints: list[ConstraintExpression]

    def __init__(self,
                 name: str,
                 params: dict[str, Param],
                 cost_function: Callable[[dict[str, ParamValue]], float],
                 constraints: list[str] = []):
        self.name = name
        self.params = params
        self.cost_function = cost_function
        self.constraints = [ConstraintExpression(s) for s in constraints]
