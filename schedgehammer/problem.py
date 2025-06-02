from abc import abstractmethod, ABC
from typing import Callable

from schedgehammer.constraint import Constraint, ConstraintExpression, Solver
from schedgehammer.param_types import Param, ParamValue


class Problem(ABC):
    name: str
    params: dict[str, Param]
    constraints: list[Constraint]
    init_solver = True

    def __init__(
        self,
        name: str,
        params: dict[str, Param],
        constraints: list[str] = [],
        init_solver = False
    ):
        self.name = name
        self.params = params
        self.constraints = [
            ConstraintExpression(s) for s in constraints
        ]
        self.init_solver = init_solver

    def get_solver(self) -> Solver:
        if not self.init_solver:
            return None
        variables = {k: v.get_value_range() for k, v in self.params.items()}
        return Solver(variables, self.constraints)

    @abstractmethod
    def cost_function(self, config: dict[str, ParamValue]) -> float:
        raise NotImplementedError()


