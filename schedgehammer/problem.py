from typing import Callable

from schedgehammer.constraint import ConstraintParser
from schedgehammer.param_types import Param, ParamValue
from schedgehammer.constraints import Constraint, Solver


class Problem:
    name: str
    params: dict[str, Param]
    constraints: list[Constraint]
    cost_function: Callable[[dict[str, ParamValue]], float]
    constraints: list[Constraint]

    def __init__(
        self,
        name: str,
        params: dict[str, Param],
        cost_function: Callable[[dict[str, ParamValue]], float],
        constraints: list[str] = [],
    ):
        self.name = name
        self.params = params
        self.cost_function = cost_function
        self.constraints = [ConstraintParser.generate(s) for s in constraints]

    def get_solver(self) -> Solver:
        variables = {k: v.get_value_range() for k, v in self.params.items()}
        return Solver(variables, self.constraints)
