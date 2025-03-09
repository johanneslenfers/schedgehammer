import copy
import math
import random
from abc import abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Iterable, Optional

from antlr4.CommonTokenStream import CommonTokenStream
from antlr4.InputStream import InputStream

from schedgehammer.param_types import ParamValue
from schedgehammer.parsing.antlr.ConstraintLexer import ConstraintLexer
from schedgehammer.parsing.antlr.ConstraintParser import ConstraintParser
from schedgehammer.parsing.visitor import (
    EvaluatingVisitor,
    FindVariablesVisitor,
    GeneratingVisitor,
)

functions = {"sqrt": math.sqrt}


class ConstraintExpression:
    expression: ConstraintParser.ExpressionContext
    fun: Callable[[dict, dict], bool]
    dependencies: set[str]

    def __init__(self, constraint: str):
        lexer = ConstraintLexer(InputStream(constraint))
        stream = CommonTokenStream(lexer)
        parser = ConstraintParser(stream)
        self.expression = parser.expression()

        if parser.getNumberOfSyntaxErrors() > 0:
            raise Exception("Could not parse constraint")

        fun_body = GeneratingVisitor().visit(self.expression)
        self.fun = eval("lambda variables, functions: " + fun_body)

        self.dependencies = FindVariablesVisitor().visit(self.expression)

    def evaluate(self, config: dict[str, ParamValue]):
        return self.fun(config, functions)

    def to_constraint(self):
        l = list(self.dependencies)
        if len(self.dependencies) == 1:
            return ConstraintUnOp(l[0], lambda x: self.evaluate({l[0]: x}))
        elif len(self.dependencies) == 2:
            return ConstraintBinOp(
                l[0], l[1], lambda x, y: self.evaluate({l[0]: x, l[1]: y})
            )
        else:
            raise Exception("Unsupported number of dependencies")


class FilterResult(Enum):
    Changed = 1
    Unchanged = 2
    Empty = 3


@dataclass
class Constraint:
    @abstractmethod
    def filter(self, variables: dict[str, list[any]]) -> FilterResult:
        raise NotImplementedError


@dataclass
class ConstraintUnOp(Constraint):
    var: str
    fn: Callable

    def filter(self, variables: dict[str, list[any]]) -> FilterResult:
        size = len(variables[self.var])

        new_domain = variables[self.var]
        new_domain = list(filter(lambda x: self.fn(x), new_domain))

        if len(new_domain) == 0:
            return FilterResult.Empty

        variables[self.var] = list(new_domain)

        if size > len(variables[self.var]):
            return FilterResult.Changed
        else:
            return FilterResult.Unchanged


@dataclass
class ConstraintBinOp(Constraint):
    var1: str
    var2: str
    fn: Callable

    def filter(self, variables: dict[str, list[any]]) -> FilterResult:
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
            return FilterResult.Empty

        variables[self.var2] = list(new_domain2)

        new_domain1 = set()
        for i in variables[self.var2]:
            new_domain1 = new_domain1.union(
                filter(lambda x: self.fn(x, i), variables[self.var1])
            )

        if len(new_domain1) == 0:
            return FilterResult.Empty

        variables[self.var1] = list(new_domain1)

        if size > len(variables[self.var1]) + len(variables[self.var2]):
            return FilterResult.Changed
        else:
            return FilterResult.Unchanged


Variables = dict[str, list[ParamValue]]


@dataclass
class Solver:
    variables: dict[str, list[ParamValue]]
    constraints: list[ConstraintBinOp]
    decision_queue: list[tuple[str, ParamValue]] = field(default_factory=list)
    exploration_function: Callable[[Iterable], ParamValue] = random.choice

    def make_decision(self, variables: Variables) -> tuple[str, ParamValue]:
        """
        Fixes a parameter to a value.
        Value is chosen based on exploration function
        """
        if len(self.decision_queue) > 0:
            var, val = self.decision_queue.pop()
            # only accept if value is still in domain
            if len(variables[var]) > 1 and val in variables[var]:
                return (var, val)

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
                if filtered == FilterResult.Empty:
                    return False
                elif filtered == FilterResult.Changed:
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
