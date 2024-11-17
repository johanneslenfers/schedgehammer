# Generated from Constraint.g4 by ANTLR 4.13.2
from antlr4 import *
if "." in __name__:
    from .ConstraintParser import ConstraintParser
else:
    from ConstraintParser import ConstraintParser

# This class defines a complete generic visitor for a parse tree produced by ConstraintParser.

class ConstraintVisitor(ParseTreeVisitor):

    # Visit a parse tree produced by ConstraintParser#BitwiseAndExpr.
    def visitBitwiseAndExpr(self, ctx:ConstraintParser.BitwiseAndExprContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ConstraintParser#ParenthesisExpr.
    def visitParenthesisExpr(self, ctx:ConstraintParser.ParenthesisExprContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ConstraintParser#FunctionExpr.
    def visitFunctionExpr(self, ctx:ConstraintParser.FunctionExprContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ConstraintParser#BitwiseOrExpr.
    def visitBitwiseOrExpr(self, ctx:ConstraintParser.BitwiseOrExprContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ConstraintParser#ComparisonExpr.
    def visitComparisonExpr(self, ctx:ConstraintParser.ComparisonExprContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ConstraintParser#LogicalNotExpr.
    def visitLogicalNotExpr(self, ctx:ConstraintParser.LogicalNotExprContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ConstraintParser#ListAccessExpr.
    def visitListAccessExpr(self, ctx:ConstraintParser.ListAccessExprContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ConstraintParser#UnarySignExpr.
    def visitUnarySignExpr(self, ctx:ConstraintParser.UnarySignExprContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ConstraintParser#LogicalAndExpr.
    def visitLogicalAndExpr(self, ctx:ConstraintParser.LogicalAndExprContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ConstraintParser#PowerExpr.
    def visitPowerExpr(self, ctx:ConstraintParser.PowerExprContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ConstraintParser#LogicalOrExpr.
    def visitLogicalOrExpr(self, ctx:ConstraintParser.LogicalOrExprContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ConstraintParser#AdditionExpr.
    def visitAdditionExpr(self, ctx:ConstraintParser.AdditionExprContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ConstraintParser#BitshiftExpr.
    def visitBitshiftExpr(self, ctx:ConstraintParser.BitshiftExprContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ConstraintParser#BitwiseNotExpr.
    def visitBitwiseNotExpr(self, ctx:ConstraintParser.BitwiseNotExprContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ConstraintParser#LiteralExpr.
    def visitLiteralExpr(self, ctx:ConstraintParser.LiteralExprContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ConstraintParser#VariableExpr.
    def visitVariableExpr(self, ctx:ConstraintParser.VariableExprContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ConstraintParser#ListExpr.
    def visitListExpr(self, ctx:ConstraintParser.ListExprContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ConstraintParser#BitwiseXorExpr.
    def visitBitwiseXorExpr(self, ctx:ConstraintParser.BitwiseXorExprContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ConstraintParser#ProductExpr.
    def visitProductExpr(self, ctx:ConstraintParser.ProductExprContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ConstraintParser#integer.
    def visitInteger(self, ctx:ConstraintParser.IntegerContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ConstraintParser#float.
    def visitFloat(self, ctx:ConstraintParser.FloatContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ConstraintParser#string.
    def visitString(self, ctx:ConstraintParser.StringContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ConstraintParser#boolean.
    def visitBoolean(self, ctx:ConstraintParser.BooleanContext):
        return self.visitChildren(ctx)



del ConstraintParser