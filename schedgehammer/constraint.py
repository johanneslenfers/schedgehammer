import math
import random
from typing import Callable

from antlr4.CommonTokenStream import CommonTokenStream
from antlr4.InputStream import InputStream

from schedgehammer.param_types import ParamValue, Param
from schedgehammer.parsing.antlr.ConstraintLexer import ConstraintLexer
from schedgehammer.parsing.antlr.ConstraintParser import ConstraintParser
from schedgehammer.parsing.visitor import (
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

SOLVER_INITIAL_TRIES = 5

class Solver:
    domains: dict[str, list[ParamValue]]
    generation_order: list[tuple[str, list[ConstraintExpression]]]

    search_tree: dict
    params: dict[str, Param]

    def __init__(self, params: dict[str, Param], constraints: list[ConstraintExpression]):
        self.generation_order = self.calculate_generation_order(params, constraints)
        self.domains = {}
        self.search_tree = {}
        self.params = params

        for param_name, constraints in self.generation_order:
            self.domains[param_name] = params[param_name].get_value_range()
            for constraint in constraints.copy():
                if len(constraint.dependencies) == 1:
                    self.domains[param_name] = list(filter(
                        lambda x: constraint.evaluate({param_name: x}),
                        self.domains[param_name]
                    ))
                    constraints.remove(constraint)

    def calculate_generation_order(self, params: dict[str, Param], constraints: list[ConstraintExpression]):
        generation_order = []
        used_params = []
        remaining_params = set(params.keys())
        remaining_constraints = set(constraints.copy())

        while len(remaining_params) > 0:
            best_param = None
            best_resolved_constraints = []

            for param in remaining_params:
                resolved_constraints = []
                for constraint in remaining_constraints:
                    if constraint.dependencies.issubset(set(used_params + [param])):
                        resolved_constraints.append(constraint)
                if len(resolved_constraints) > len(best_resolved_constraints):
                    best_resolved_constraints = resolved_constraints
                    best_param = param

            if best_param is None:
                best_param = list(remaining_params)[0]

            generation_order.append((best_param, best_resolved_constraints))
            remaining_params.remove(best_param)
            used_params.append(best_param)
            remaining_constraints.difference_update(best_resolved_constraints)
        return generation_order

    def search_recursively(self, config, search_branch, depth = 0, around_config = None) -> bool:
        if depth >= len(self.generation_order):
            return True

        param_name, constraints = self.generation_order[depth]

        # TODO:  Probably do this more for params for which we don't have a domain.
        for i in range(SOLVER_INITIAL_TRIES):
            # Choose random value.
            if param_name in self.domains and around_config is None:
                config[param_name] = random.choice(self.domains[param_name])
            else:
                config[param_name] = self.params[param_name].choose_random(around_config[param_name])

            config_value_str = str(config[param_name])  # Should this be hashed (cheaply) if string is long?

            if config_value_str in search_branch and search_branch[config_value_str] is None:
                continue

            constraints_fulfilled = all(map(lambda constraint: constraint.evaluate(config), constraints))
            if not constraints_fulfilled:
                search_branch[config_value_str] = None
                continue

            if config_value_str not in search_branch:
                search_branch[config_value_str] = {}

            if self.search_recursively(config, search_branch[config_value_str], depth + 1, around_config):
                return True
            else:
                search_branch[config_value_str] = None

        def test_value(v):
            config[param_name] = v
            return all(map(lambda constraint: constraint.evaluate(config), constraints))

        if param_name in self.domains:
            if len(constraints) > 0:
                domain = list(filter(test_value, self.domains[param_name]))
            else:
                domain = self.domains[param_name].copy()
            while len(domain) > 0:
                if around_config is None:
                    value = random.choice(domain)
                else:
                    value = self.params[param_name].choose_random_around_in(around_config[param_name], domain)
                str_value = str(value)
                config[param_name] = value
                domain.remove(value)
                if str_value in search_branch and search_branch[str_value] is None:
                    continue

                if str_value not in search_branch:
                    search_branch[str_value] = {}

                if self.search_recursively(config, search_branch[str_value], depth + 1, around_config):
                    return True
                else:
                    search_branch[str_value] = None

        return False

    def solve(self, around_config = None):
        config = {}
        while not self.search_recursively(config, self.search_tree, 0, around_config):  # TODO: Change
            pass
        return config
