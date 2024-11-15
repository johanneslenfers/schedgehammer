import copy
import random
from dataclasses import dataclass
from typing import Callable, Iterable, Optional
from param_types import ParamValue


@dataclass
class Constraint:
    var1: str
    var2: str
    fn: Callable

    def filter(self, variables: dict[str, list[any]]) -> Optional[bool]:
        """
        Filters variable domains.
        Returns True if changed, False if no valid values remain,
        None if nothing changed
        """
        size = len(variables[self.var1]) + len(variables[self.var2])

        new_domain2 = set()
        for i in variables[self.var1]:
            new_domain2 = new_domain2.union(
                filter(lambda x: self.fn(i, x), variables[self.var2])
            )

        if len(new_domain2) == 0:
            return False

        variables[self.var2] = list(new_domain2)

        new_domain1 = set()
        for i in variables[self.var2]:
            new_domain1 = new_domain1.union(
                filter(lambda x: self.fn(x, i), variables[self.var1])
            )

        if len(new_domain1) == 0:
            return False

        variables[self.var1] = list(new_domain1)

        if size > len(variables[self.var1]) + len(variables[self.var2]):
            return True
        else:
            return None


Variables = dict[str, list[ParamValue]]


@dataclass
class Solver:
    variables: dict[str, list[any]]
    constraints: list[Constraint]
    exploration_function: Callable[[Iterable], ParamValue] = random.choice

    def make_decision(self, variables: Variables) -> tuple[str, ParamValue]:
        """
        Fixes a parameter to a value.
        Value is chosen based on exploration function
        """
        var, domain = next(
            filter(lambda x: len(x[1]) > 1, variables.items()), (None, None)
        )

        if var is not None:
            return var, self.exploration_function(domain)
        else:
            return None, None

    def fix_point(self, variables: Variables) -> bool:
        """
        Returns True, if applying any constraint does not change the search space
        """
        while True:
            fltrs = False
            for c in self.constraints:
                filtered = c.filter(variables)
                if filtered is False:
                    return False
                elif filtered is True:
                    fltrs |= True
            if not fltrs:
                return True

    def apply_decision(
        self, variables: Variables, var: str, val: ParamValue, apply: bool
    ):
        """
        If apply is True, fix parameter to single value, else remove
        value from search space
        """
        c = copy.deepcopy(variables)
        c[var] = list(filter(lambda x: apply == (x == val), c[var]))
        yield from self.dfs(c)

    def dfs(self, variables: Variables):
        """
        Generate valid configurations, using dfs
        """
        if self.fix_point(variables):
            var, val = self.make_decision(variables)
            if var is None:
                yield {k: min(v) for k, v in variables.items()}
            else:
                yield from self.apply_decision(variables, var, val, True)
                yield from self.apply_decision(variables, var, val, False)

    def solve(self):
        yield from self.dfs(self.variables)


if __name__ == "__main__":
    variables = {"a": [1, 2, 3], "b": [1, 2, 3]}
    g = Constraint("a", "b", lambda x, y: x == y)
    solver = Solver(variables, [g], random.choice)
    solutions = list(solver.solve())
    print(f"there are {len(solutions)} solutions")
    print(solutions)
