from typing import Callable

from schedgehammer.constraint import Constraint, ConstraintExpression, Solver
from schedgehammer.param_types import Param, ParamValue


class Problem:
    name: str
    params: dict[str, Param]
    cost_function: Callable[[dict[str, ParamValue]], float]
    constraints: list[Constraint]
    init_solver = True

    def __init__(
        self,
        name: str,
        params: dict[str, Param],
        cost_function: Callable[[dict[str, ParamValue]], float],
        constraints: list[str] = [],
        init_solver = False
    ):
        self.name = name
        self.params = params
        self.cost_function = cost_function
        self.constraints = [
            ConstraintExpression(s).to_constraint() for s in constraints
        ]
        self.init_solver = init_solver

    def get_solver(self) -> Solver:
        if not self.init_solver:
            return None
        variables = {k: v.get_value_range() for k, v in self.params.items()}
        return Solver(variables, self.constraints)
