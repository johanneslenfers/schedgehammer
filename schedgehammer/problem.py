from typing import Callable, Optional

from schedgehammer.constraint import Constraint, ConstraintExpression, Solver, Solver2
from schedgehammer.param_types import Param, ParamValue


class Problem:
    name: str
    params: dict[str, Param]
    cost_function: Callable[[dict[str, ParamValue]], float]
    constraints: list[Constraint]
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
        self.constraints = [
            ce.to_constraint() for ce in self.constraint_expressions
        ]

    def get_solver(self) -> Solver2:
        return Solver2(self.params, self.constraint_expressions)
