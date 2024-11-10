import math
from typing import Callable

from antlr4.CommonTokenStream import CommonTokenStream
from antlr4.InputStream import InputStream

from schedgehammer.param_types import ParamValue
from schedgehammer.parsing.antlr.ConstraintLexer import ConstraintLexer
from schedgehammer.parsing.antlr.ConstraintParser import ConstraintParser
from schedgehammer.parsing.visitor import (
    FindVariablesVisitor,
    EvaluatingVisitor,
    GeneratingVisitor,
)

functions = {"sqrt": math.sqrt}


class ConstraintParser:
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

    def generate(self):
        return GeneratingVisitor().visit(self.expression)

    def evaluate(self, config: dict[str, ParamValue]):
        return self.fun(config, functions)
