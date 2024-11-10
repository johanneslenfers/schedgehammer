import copy
import random
from dataclasses import dataclass
from typing import Callable, Iterable


@dataclass
class Constraint:
    pass


@dataclass
class LessThan(Constraint):
    var1: str
    var2: str

    def filter(self, variables: dict[str, list[any]]):
        new_domain1 = list(
            filter(lambda x: x < max(variables[self.var2]), variables[self.var1])
        )
        new_domain2 = list(
            filter(lambda x: x > min(variables[self.var1]), variables[self.var2])
        )

        if len(new_domain1) == 0 or len(new_domain2) == 0:
            return False

        variables[self.var1] = new_domain1
        variables[self.var2] = new_domain2

        if len(new_domain1) < len(variables[self.var1]) or len(
            variables[self.var2]
        ) < len(new_domain2):
            return True

        return None


@dataclass
class NotEqual(Constraint):
    var1: str
    var2: str

    def filter(self, variables: dict[str, list[any]]):
        size = len(variables[self.var1]) + len(variables[self.var2])
        if len(variables[self.var1]) == 1:
            new_domain2 = list(
                filter(lambda x: x != variables[self.var1][0], variables[self.var2])
            )
            if len(new_domain2) == 0:
                return False
            variables[self.var2] = new_domain2
        if len(variables[self.var2]) == 1:
            new_domain1 = list(
                filter(lambda x: x != variables[self.var2][0], variables[self.var1])
            )
            if len(new_domain1) == 0:
                return False
            variables[self.var1] = new_domain1

        if size > len(variables[self.var1]) + len(variables[self.var2]):
            return True
        else:
            return None


@dataclass
class Solver:
    variables: dict[str, list[any]]
    constraints: list[Constraint]
    exploration_function: Callable[[Iterable], any] = random.choice

    def make_decision(self, variables):
        var, domain = next(
            filter(lambda x: len(x[1]) > 1, variables.items()), (None, None)
        )

        if var is not None:
            return var, self.exploration_function(domain)
        else:
            return None, None

    def fix_point(self, variables):
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

    def apply_decision(self, variables, var, val, apply):
        c = copy.deepcopy(variables)
        c[var] = list(filter(lambda x: apply == (x == val), c[var]))
        yield from self.dfs(c)

    def dfs(self, variables):
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
    ne = NotEqual("a", "b")
    lt = LessThan("a", "b")
    solver = Solver(variables, [lt], min)
    solutions = list(solver.solve())
    print(f"there are {len(solutions)} solutions")
    print(solutions)
