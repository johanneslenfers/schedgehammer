from typing import Callable

from schedgehammer.constraint import ConstraintExpression, Solver
from schedgehammer.param_types import Param, ParamValue


class Problem:
    name: str
    params: dict[str, Param]
    cost_function: Callable[[dict[str, ParamValue]], float]
    constraint_expressions: list[ConstraintExpression]

    def __init__(
        self,
        name: str,
        params: dict[str, Param],
        cost_function: Callable[[dict[str, ParamValue]], float],
        constraints: list[str],
    ):
        self.name = name
        self.params = params
        self.cost_function = cost_function
        self.constraint_expressions =  [
            ConstraintExpression(s) for s in constraints
        ]

    def get_solver(self) -> Solver:
        return Solver(self.params, self.constraint_expressions)
