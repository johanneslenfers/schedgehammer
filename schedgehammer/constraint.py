import math

from antlr4.CommonTokenStream import CommonTokenStream
from antlr4.InputStream import InputStream

from schedgehammer.param_types import ParamValue
from schedgehammer.parsing.antlr.ConstraintLexer import ConstraintLexer
from schedgehammer.parsing.antlr.ConstraintParser import ConstraintParser
from schedgehammer.parsing.visitor import FindVariablesVisitor, EvaluatingVisitor

functions = {
    'sqrt': math.sqrt
}

class Constraint:

    expression: ConstraintParser.ExpressionContext
    dependencies: set[str]

    def __init__(self, constraint: str):
        lexer = ConstraintLexer(InputStream(constraint))
        stream = CommonTokenStream(lexer)
        parser = ConstraintParser(stream)
        self.expression = parser.expression()

        if parser.getNumberOfSyntaxErrors() > 0:
            raise Exception("Could not parse constraint")

        self.dependencies = FindVariablesVisitor().visit(self.expression)

    def evaluate(self, config: dict[str, ParamValue]):
        return EvaluatingVisitor(config, functions).visit(self.expression)
