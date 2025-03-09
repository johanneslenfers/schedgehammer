import json
import math
from dataclasses import dataclass

from schedgehammer.parsing.antlr.ConstraintParser import ConstraintParser
from schedgehammer.parsing.antlr.ConstraintVisitor import ConstraintVisitor


@dataclass
class EvaluatingVisitor(ConstraintVisitor):
    variables: dict
    functions: dict

    # Visit a parse tree produced by ConstraintParser#BitwiseAndExpr.
    def visitBitwiseAndExpr(self, ctx: ConstraintParser.BitwiseAndExprContext):
        assert ctx.getChildCount() == 3
        return self.visit(ctx.getChild(0)) ^ self.visit(ctx.getChild(2))

    # Visit a parse tree produced by ConstraintParser#ParenthesisExpr.
    def visitParenthesisExpr(self, ctx: ConstraintParser.ParenthesisExprContext):
        assert ctx.getChildCount() == 3
        return self.visit(ctx.getChild(1))

    # Visit a parse tree produced by ConstraintParser#FunctionExpr.
    def visitFunctionExpr(self, ctx: ConstraintParser.FunctionExprContext):
        assert ctx.getChildCount() == 3 or (ctx.getChildCount() - 4) % 2 == 0
        argument_length = math.ceil((ctx.getChildCount() - 3) / 2)
        arguments = [
            self.visit(ctx.getChild(i * 2 + 2)) for i in range(argument_length)
        ]
        return self.functions[ctx.getChild(0).getText()](*arguments)

    # Visit a parse tree produced by ConstraintParser#BitwiseOrExpr.
    def visitBitwiseOrExpr(self, ctx: ConstraintParser.BitwiseOrExprContext):
        assert ctx.getChildCount() == 3
        return self.visit(ctx.getChild(0)) | self.visit(ctx.getChild(2))

    # Visit a parse tree produced by ConstraintParser#ComparisonExpr.
    def visitComparisonExpr(self, ctx: ConstraintParser.ComparisonExprContext):
        assert ctx.getChildCount() == 3
        c0 = self.visit(ctx.getChild(0))
        c1 = self.visit(ctx.getChild(2))
        match ctx.getChild(1).getText():
            case "==":
                return c0 == c1
            case "!=":
                return c0 != c1
            case "<":
                return c0 < c1
            case "<=":
                return c0 <= c1
            case ">":
                return c0 > c1
            case ">=":
                return c0 >= c1
            case _:
                raise Exception("Invalid Comparison Expression")

    # Visit a parse tree produced by ConstraintParser#LogicalNotExpr.
    def visitLogicalNotExpr(self, ctx: ConstraintParser.LogicalNotExprContext):
        assert ctx.getChildCount() == 2
        return not self.visit(ctx.getChild(1))

    # Visit a parse tree produced by ConstraintParser#ListAccessExpr.
    def visitListAccessExpr(self, ctx: ConstraintParser.ListAccessExprContext):
        assert ctx.getChildCount() == 4
        return self.visit(ctx.getChild(0))[self.visit(ctx.getChild(2))]

    # Visit a parse tree produced by ConstraintParser#UnarySignExpr.
    def visitUnarySignExpr(self, ctx: ConstraintParser.UnarySignExprContext):
        assert ctx.getChildCount() == 2
        c = self.visit(ctx.getChild(1))
        match ctx.getChild(0).getText():
            case "-":
                return -c
            case "+":
                return +c
            case _:
                raise Exception("Invalid Unary Sign Expression")

    # Visit a parse tree produced by ConstraintParser#LogicalAndExpr.
    def visitLogicalAndExpr(self, ctx: ConstraintParser.LogicalAndExprContext):
        return self.visit(ctx.getChild(0)) and self.visit(ctx.getChild(2))

    # Visit a parse tree produced by ConstraintParser#PowerExpr.
    def visitPowerExpr(self, ctx: ConstraintParser.PowerExprContext):
        return self.visit(ctx.getChild(0)) ** self.visit(ctx.getChild(2))

    # Visit a parse tree produced by ConstraintParser#LogicalOrExpr.
    def visitLogicalOrExpr(self, ctx: ConstraintParser.LogicalOrExprContext):
        return self.visit(ctx.getChild(0)) or self.visit(ctx.getChild(2))

    # Visit a parse tree produced by ConstraintParser#AdditionExpr.
    def visitAdditionExpr(self, ctx: ConstraintParser.AdditionExprContext):
        assert ctx.getChildCount() == 3
        c0 = self.visit(ctx.getChild(0))
        c1 = self.visit(ctx.getChild(2))
        match ctx.getChild(1).getText():
            case "+":
                return c0 + c1
            case "-":
                return c0 - c1
            case _:
                raise Exception("Invalid Addition Expression")

    # Visit a parse tree produced by ConstraintParser#BitshiftExpr.
    def visitBitshiftExpr(self, ctx: ConstraintParser.BitshiftExprContext):
        assert ctx.getChildCount() == 3
        c0 = self.visit(ctx.getChild(0))
        c1 = self.visit(ctx.getChild(2))
        match ctx.getChild(1).getText():
            case "<<":
                return c0 << c1
            case ">>":
                return c0 >> c1
            case _:
                raise Exception("Invalid Bitshift Expression")

    # Visit a parse tree produced by ConstraintParser#BitwiseNotExpr.
    def visitBitwiseNotExpr(self, ctx: ConstraintParser.BitwiseNotExprContext):
        return ~self.visit(ctx.getChild(1))

    # Visit a parse tree produced by ConstraintParser#LiteralExpr.
    def visitLiteralExpr(self, ctx: ConstraintParser.LiteralExprContext):
        assert ctx.getChildCount() == 1
        return self.visit(ctx.getChild(0))

    # Visit a parse tree produced by ConstraintParser#VariableExpr.
    def visitVariableExpr(self, ctx: ConstraintParser.VariableExprContext):
        return self.variables[ctx.getText()]

    # Visit a parse tree produced by ConstraintParser#ListExpr.
    def visitListExpr(self, ctx: ConstraintParser.ListExprContext):
        raise NotImplementedError()

    # Visit a parse tree produced by ConstraintParser#BitwiseXorExpr.
    def visitBitwiseXorExpr(self, ctx: ConstraintParser.BitwiseXorExprContext):
        return self.visit(ctx.getChild(0)) ^ self.visit(ctx.getChild(2))

    # Visit a parse tree produced by ConstraintParser#ProductExpr.
    def visitProductExpr(self, ctx: ConstraintParser.ProductExprContext):
        assert ctx.getChildCount() == 3
        c0 = self.visit(ctx.getChild(0))
        c1 = self.visit(ctx.getChild(2))
        match ctx.getChild(1).getText():
            case "*":
                return c0 * c1
            case "/":
                return c0 / c1
            case "%":
                return c0 % c1
            case "//":
                return c0 // c1
            case _:
                raise Exception("Invalid Comparison Expression")

    # Visit a parse tree produced by ConstraintParser#integer.
    def visitInteger(self, ctx: ConstraintParser.IntegerContext):
        return int(ctx.getText())

    # Visit a parse tree produced by ConstraintParser#float.
    def visitFloat(self, ctx: ConstraintParser.FloatContext):
        return float(ctx.getText())

    # Visit a parse tree produced by ConstraintParser#string.
    def visitString(self, ctx: ConstraintParser.StringContext):
        return ctx.getText()[1:-1]

    # Visit a parse tree produced by ConstraintParser#boolean.
    def visitBoolean(self, ctx: ConstraintParser.BooleanContext):
        return ctx.getText() in ["True"]


class GeneratingVisitor(ConstraintVisitor):
    # Visit a parse tree produced by ConstraintParser#BitwiseAndExpr.
    def visitBitwiseAndExpr(self, ctx: ConstraintParser.BitwiseAndExprContext):
        assert ctx.getChildCount() == 3
        return f"({self.visit(ctx.getChild(0))} ^ {self.visit(ctx.getChild(2))})"

    # Visit a parse tree produced by ConstraintParser#ParenthesisExpr.
    def visitParenthesisExpr(self, ctx: ConstraintParser.ParenthesisExprContext):
        assert ctx.getChildCount() == 3
        return f"({self.visit(ctx.getChild(1))})"

    # Visit a parse tree produced by ConstraintParser#FunctionExpr.
    def visitFunctionExpr(self, ctx: ConstraintParser.FunctionExprContext):
        assert ctx.getChildCount() == 3 or (ctx.getChildCount() - 4) % 2 == 0
        argument_length = math.ceil((ctx.getChildCount() - 3) / 2)
        arguments = [
            self.visit(ctx.getChild(i * 2 + 2)) for i in range(argument_length)
        ]
        arguments_str = ", ".join(arguments)
        function_name = ctx.getChild(0).getText()
        assert isinstance(function_name, str)
        escaped_function_name = json.dumps(function_name)
        return f"(functions[{escaped_function_name}]({arguments_str}))"

    # Visit a parse tree produced by ConstraintParser#BitwiseOrExpr.
    def visitBitwiseOrExpr(self, ctx: ConstraintParser.BitwiseOrExprContext):
        assert ctx.getChildCount() == 3
        return f"({self.visit(ctx.getChild(0))} | {self.visit(ctx.getChild(2))})"

    # Visit a parse tree produced by ConstraintParser#ComparisonExpr.
    def visitComparisonExpr(self, ctx: ConstraintParser.ComparisonExprContext):
        assert ctx.getChildCount() == 3
        c0 = self.visit(ctx.getChild(0))
        c1 = self.visit(ctx.getChild(2))
        match ctx.getChild(1).getText():
            case "==":
                return f"({c0} == {c1})"
            case "!=":
                return f"({c0} != {c1})"
            case "<":
                return f"({c0} < {c1})"
            case "<=":
                return f"({c0} <= {c1})"
            case ">":
                return f"({c0} > {c1})"
            case ">=":
                return f"({c0} >= {c1})"
            case _:
                raise Exception("Invalid Comparison Expression")

    # Visit a parse tree produced by ConstraintParser#LogicalNotExpr.
    def visitLogicalNotExpr(self, ctx: ConstraintParser.LogicalNotExprContext):
        assert ctx.getChildCount() == 2
        return f"(not {self.visit(ctx.getChild(1))})"

    # Visit a parse tree produced by ConstraintParser#ListAccessExpr.
    def visitListAccessExpr(self, ctx: ConstraintParser.ListAccessExprContext):
        assert ctx.getChildCount() == 4
        return f"({self.visit(ctx.getChild(0))}[{self.visit(ctx.getChild(2))}])"

    # Visit a parse tree produced by ConstraintParser#UnarySignExpr.
    def visitUnarySignExpr(self, ctx: ConstraintParser.UnarySignExprContext):
        assert ctx.getChildCount() == 2
        c = self.visit(ctx.getChild(1))
        match ctx.getChild(0).getText():
            case "-":
                return f"(-{c})"
            case "+":
                return f"(+{c})"
            case _:
                raise Exception("Invalid Unary Sign Expression")

    # Visit a parse tree produced by ConstraintParser#LogicalAndExpr.
    def visitLogicalAndExpr(self, ctx: ConstraintParser.LogicalAndExprContext):
        return f"({self.visit(ctx.getChild(0))} and {self.visit(ctx.getChild(2))})"

    # Visit a parse tree produced by ConstraintParser#PowerExpr.
    def visitPowerExpr(self, ctx: ConstraintParser.PowerExprContext):
        return f"({self.visit(ctx.getChild(0))} ** {self.visit(ctx.getChild(2))})"

    # Visit a parse tree produced by ConstraintParser#LogicalOrExpr.
    def visitLogicalOrExpr(self, ctx: ConstraintParser.LogicalOrExprContext):
        return f"({self.visit(ctx.getChild(0))} or {self.visit(ctx.getChild(2))})"

    # Visit a parse tree produced by ConstraintParser#AdditionExpr.
    def visitAdditionExpr(self, ctx: ConstraintParser.AdditionExprContext):
        assert ctx.getChildCount() == 3
        c0 = self.visit(ctx.getChild(0))
        c1 = self.visit(ctx.getChild(2))
        match ctx.getChild(1).getText():
            case "+":
                return f"({c0} + {c1})"
            case "-":
                return f"({c0} - {c1})"
            case _:
                raise Exception("Invalid Addition Expression")

    # Visit a parse tree produced by ConstraintParser#BitshiftExpr.
    def visitBitshiftExpr(self, ctx: ConstraintParser.BitshiftExprContext):
        assert ctx.getChildCount() == 3
        c0 = self.visit(ctx.getChild(0))
        c1 = self.visit(ctx.getChild(2))
        match ctx.getChild(1).getText():
            case "<<":
                return f"({c0} << {c1})"
            case ">>":
                return f"({c0} >> {c1})"
            case _:
                raise Exception("Invalid Bitshift Expression")

    # Visit a parse tree produced by ConstraintParser#BitwiseNotExpr.
    def visitBitwiseNotExpr(self, ctx: ConstraintParser.BitwiseNotExprContext):
        return f"(~{self.visit(ctx.getChild(1))})"

    # Visit a parse tree produced by ConstraintParser#LiteralExpr.
    def visitLiteralExpr(self, ctx: ConstraintParser.LiteralExprContext):
        assert ctx.getChildCount() == 1
        return self.visit(ctx.getChild(0))

    # Visit a parse tree produced by ConstraintParser#VariableExpr.
    def visitVariableExpr(self, ctx: ConstraintParser.VariableExprContext):
        variable_name = ctx.getText()
        escaped_variable_name = json.dumps(variable_name)
        return f"(variables[{escaped_variable_name}])"

    # Visit a parse tree produced by ConstraintParser#ListExpr.
    def visitListExpr(self, ctx: ConstraintParser.ListExprContext):
        raise NotImplementedError()

    # Visit a parse tree produced by ConstraintParser#BitwiseXorExpr.
    def visitBitwiseXorExpr(self, ctx: ConstraintParser.BitwiseXorExprContext):
        return f"({self.visit(ctx.getChild(0))} ^ {self.visit(ctx.getChild(2))})"

    # Visit a parse tree produced by ConstraintParser#ProductExpr.
    def visitProductExpr(self, ctx: ConstraintParser.ProductExprContext):
        assert ctx.getChildCount() == 3
        c0 = self.visit(ctx.getChild(0))
        c1 = self.visit(ctx.getChild(2))
        match ctx.getChild(1).getText():
            case "*":
                return f"({c0} * {c1})"
            case "/":
                return f"({c0} / {c1})"
            case "%":
                return f"({c0} % {c1})"
            case "//":
                return f"({c0} // {c1})"
            case _:
                raise Exception("Invalid Comparison Expression")

    # Visit a parse tree produced by ConstraintParser#integer.
    def visitInteger(self, ctx: ConstraintParser.IntegerContext):
        return str(int(ctx.getText()))

    # Visit a parse tree produced by ConstraintParser#float.
    def visitFloat(self, ctx: ConstraintParser.FloatContext):
        return str(float(ctx.getText()))

    # Visit a parse tree produced by ConstraintParser#string.
    def visitString(self, ctx: ConstraintParser.StringContext):
        return json.dumps(ctx.getText()[1:-1])

    # Visit a parse tree produced by ConstraintParser#boolean.
    def visitBoolean(self, ctx: ConstraintParser.BooleanContext):
        return str(ctx.getText() in ["True"])


class FindVariablesVisitor(ConstraintVisitor):

    def defaultResult(self):
        return set()

    def aggregateResult(self, aggregate: set, next_result: set):
        return aggregate.union(next_result)

    def visitVariableExpr(self, ctx: ConstraintParser.VariableExprContext):
        return {ctx.getChild(0).getText()}
