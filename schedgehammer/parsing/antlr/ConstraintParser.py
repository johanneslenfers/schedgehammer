# Generated from Constraint.g4 by ANTLR 4.13.2
# encoding: utf-8
from antlr4 import *
from io import StringIO
import sys
if sys.version_info[1] > 5:
	from typing import TextIO
else:
	from typing.io import TextIO

def serializedATN():
    return [
        4,1,35,104,2,0,7,0,2,1,7,1,2,2,7,2,2,3,7,3,2,4,7,4,1,0,1,0,1,0,1,
        0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,5,0,28,8,0,
        10,0,12,0,31,9,0,3,0,33,8,0,1,0,1,0,1,0,1,0,1,0,5,0,40,8,0,10,0,
        12,0,43,9,0,3,0,45,8,0,1,0,1,0,1,0,1,0,1,0,3,0,52,8,0,3,0,54,8,0,
        1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,
        1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,
        1,0,1,0,1,0,5,0,91,8,0,10,0,12,0,94,9,0,1,1,1,1,1,2,1,2,1,3,1,3,
        1,4,1,4,1,4,0,1,0,5,0,2,4,6,8,0,6,1,0,7,8,1,0,9,11,1,0,12,13,1,0,
        17,22,1,0,24,25,1,0,26,27,123,0,53,1,0,0,0,2,95,1,0,0,0,4,97,1,0,
        0,0,6,99,1,0,0,0,8,101,1,0,0,0,10,11,6,0,-1,0,11,12,5,1,0,0,12,13,
        3,0,0,0,13,14,5,2,0,0,14,54,1,0,0,0,15,16,5,6,0,0,16,54,3,0,0,16,
        17,18,7,0,0,0,18,54,3,0,0,15,19,20,5,23,0,0,20,54,3,0,0,7,21,54,
        5,34,0,0,22,23,5,34,0,0,23,32,5,1,0,0,24,29,3,0,0,0,25,26,5,28,0,
        0,26,28,3,0,0,0,27,25,1,0,0,0,28,31,1,0,0,0,29,27,1,0,0,0,29,30,
        1,0,0,0,30,33,1,0,0,0,31,29,1,0,0,0,32,24,1,0,0,0,32,33,1,0,0,0,
        33,34,1,0,0,0,34,54,5,2,0,0,35,44,5,3,0,0,36,41,3,0,0,0,37,38,5,
        28,0,0,38,40,3,0,0,0,39,37,1,0,0,0,40,43,1,0,0,0,41,39,1,0,0,0,41,
        42,1,0,0,0,42,45,1,0,0,0,43,41,1,0,0,0,44,36,1,0,0,0,44,45,1,0,0,
        0,45,46,1,0,0,0,46,54,5,4,0,0,47,52,3,2,1,0,48,52,3,4,2,0,49,52,
        3,6,3,0,50,52,3,8,4,0,51,47,1,0,0,0,51,48,1,0,0,0,51,49,1,0,0,0,
        51,50,1,0,0,0,52,54,1,0,0,0,53,10,1,0,0,0,53,15,1,0,0,0,53,17,1,
        0,0,0,53,19,1,0,0,0,53,21,1,0,0,0,53,22,1,0,0,0,53,35,1,0,0,0,53,
        51,1,0,0,0,54,92,1,0,0,0,55,56,10,17,0,0,56,57,5,5,0,0,57,91,3,0,
        0,18,58,59,10,14,0,0,59,60,7,1,0,0,60,91,3,0,0,15,61,62,10,13,0,
        0,62,63,7,0,0,0,63,91,3,0,0,14,64,65,10,12,0,0,65,66,7,2,0,0,66,
        91,3,0,0,13,67,68,10,11,0,0,68,69,5,14,0,0,69,91,3,0,0,12,70,71,
        10,10,0,0,71,72,5,15,0,0,72,91,3,0,0,11,73,74,10,9,0,0,74,75,5,16,
        0,0,75,91,3,0,0,10,76,77,10,8,0,0,77,78,7,3,0,0,78,91,3,0,0,9,79,
        80,10,6,0,0,80,81,7,4,0,0,81,91,3,0,0,7,82,83,10,5,0,0,83,84,7,5,
        0,0,84,91,3,0,0,6,85,86,10,18,0,0,86,87,5,3,0,0,87,88,3,0,0,0,88,
        89,5,4,0,0,89,91,1,0,0,0,90,55,1,0,0,0,90,58,1,0,0,0,90,61,1,0,0,
        0,90,64,1,0,0,0,90,67,1,0,0,0,90,70,1,0,0,0,90,73,1,0,0,0,90,76,
        1,0,0,0,90,79,1,0,0,0,90,82,1,0,0,0,90,85,1,0,0,0,91,94,1,0,0,0,
        92,90,1,0,0,0,92,93,1,0,0,0,93,1,1,0,0,0,94,92,1,0,0,0,95,96,5,29,
        0,0,96,3,1,0,0,0,97,98,5,31,0,0,98,5,1,0,0,0,99,100,5,30,0,0,100,
        7,1,0,0,0,101,102,5,32,0,0,102,9,1,0,0,0,8,29,32,41,44,51,53,90,
        92
    ]

class ConstraintParser ( Parser ):

    grammarFileName = "Constraint.g4"

    atn = ATNDeserializer().deserialize(serializedATN())

    decisionsToDFA = [ DFA(ds, i) for i, ds in enumerate(atn.decisionToState) ]

    sharedContextCache = PredictionContextCache()

    literalNames = [ "<INVALID>", "'('", "')'", "'['", "']'", "'**'", "'~'", 
                     "'+'", "'-'", "'*'", "'/'", "'%'", "'<<'", "'>>'", 
                     "'&'", "'^'", "'|'", "'=='", "'!='", "'>'", "'>='", 
                     "'<'", "'<='", "'not'", "'and'", "'&&'", "'or'", "'||'", 
                     "','" ]

    symbolicNames = [ "<INVALID>", "<INVALID>", "<INVALID>", "<INVALID>", 
                      "<INVALID>", "<INVALID>", "<INVALID>", "<INVALID>", 
                      "<INVALID>", "<INVALID>", "<INVALID>", "<INVALID>", 
                      "<INVALID>", "<INVALID>", "<INVALID>", "<INVALID>", 
                      "<INVALID>", "<INVALID>", "<INVALID>", "<INVALID>", 
                      "<INVALID>", "<INVALID>", "<INVALID>", "<INVALID>", 
                      "<INVALID>", "<INVALID>", "<INVALID>", "<INVALID>", 
                      "<INVALID>", "INTEGER", "STRING", "FLOAT", "BOOLEAN", 
                      "SIGN", "IDENTIFIER", "IGNORE" ]

    RULE_expression = 0
    RULE_integer = 1
    RULE_float = 2
    RULE_string = 3
    RULE_boolean = 4

    ruleNames =  [ "expression", "integer", "float", "string", "boolean" ]

    EOF = Token.EOF
    T__0=1
    T__1=2
    T__2=3
    T__3=4
    T__4=5
    T__5=6
    T__6=7
    T__7=8
    T__8=9
    T__9=10
    T__10=11
    T__11=12
    T__12=13
    T__13=14
    T__14=15
    T__15=16
    T__16=17
    T__17=18
    T__18=19
    T__19=20
    T__20=21
    T__21=22
    T__22=23
    T__23=24
    T__24=25
    T__25=26
    T__26=27
    T__27=28
    INTEGER=29
    STRING=30
    FLOAT=31
    BOOLEAN=32
    SIGN=33
    IDENTIFIER=34
    IGNORE=35

    def __init__(self, input:TokenStream, output:TextIO = sys.stdout):
        super().__init__(input, output)
        self.checkVersion("4.13.2")
        self._interp = ParserATNSimulator(self, self.atn, self.decisionsToDFA, self.sharedContextCache)
        self._predicates = None




    class ExpressionContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser


        def getRuleIndex(self):
            return ConstraintParser.RULE_expression

     
        def copyFrom(self, ctx:ParserRuleContext):
            super().copyFrom(ctx)


    class BitwiseAndExprContext(ExpressionContext):

        def __init__(self, parser, ctx:ParserRuleContext): # actually a ConstraintParser.ExpressionContext
            super().__init__(parser)
            self.copyFrom(ctx)

        def expression(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(ConstraintParser.ExpressionContext)
            else:
                return self.getTypedRuleContext(ConstraintParser.ExpressionContext,i)


        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterBitwiseAndExpr" ):
                listener.enterBitwiseAndExpr(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitBitwiseAndExpr" ):
                listener.exitBitwiseAndExpr(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitBitwiseAndExpr" ):
                return visitor.visitBitwiseAndExpr(self)
            else:
                return visitor.visitChildren(self)


    class ParenthesisExprContext(ExpressionContext):

        def __init__(self, parser, ctx:ParserRuleContext): # actually a ConstraintParser.ExpressionContext
            super().__init__(parser)
            self.copyFrom(ctx)

        def expression(self):
            return self.getTypedRuleContext(ConstraintParser.ExpressionContext,0)


        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterParenthesisExpr" ):
                listener.enterParenthesisExpr(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitParenthesisExpr" ):
                listener.exitParenthesisExpr(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitParenthesisExpr" ):
                return visitor.visitParenthesisExpr(self)
            else:
                return visitor.visitChildren(self)


    class FunctionExprContext(ExpressionContext):

        def __init__(self, parser, ctx:ParserRuleContext): # actually a ConstraintParser.ExpressionContext
            super().__init__(parser)
            self.copyFrom(ctx)

        def IDENTIFIER(self):
            return self.getToken(ConstraintParser.IDENTIFIER, 0)
        def expression(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(ConstraintParser.ExpressionContext)
            else:
                return self.getTypedRuleContext(ConstraintParser.ExpressionContext,i)


        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterFunctionExpr" ):
                listener.enterFunctionExpr(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitFunctionExpr" ):
                listener.exitFunctionExpr(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitFunctionExpr" ):
                return visitor.visitFunctionExpr(self)
            else:
                return visitor.visitChildren(self)


    class BitwiseOrExprContext(ExpressionContext):

        def __init__(self, parser, ctx:ParserRuleContext): # actually a ConstraintParser.ExpressionContext
            super().__init__(parser)
            self.copyFrom(ctx)

        def expression(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(ConstraintParser.ExpressionContext)
            else:
                return self.getTypedRuleContext(ConstraintParser.ExpressionContext,i)


        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterBitwiseOrExpr" ):
                listener.enterBitwiseOrExpr(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitBitwiseOrExpr" ):
                listener.exitBitwiseOrExpr(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitBitwiseOrExpr" ):
                return visitor.visitBitwiseOrExpr(self)
            else:
                return visitor.visitChildren(self)


    class ComparisonExprContext(ExpressionContext):

        def __init__(self, parser, ctx:ParserRuleContext): # actually a ConstraintParser.ExpressionContext
            super().__init__(parser)
            self.copyFrom(ctx)

        def expression(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(ConstraintParser.ExpressionContext)
            else:
                return self.getTypedRuleContext(ConstraintParser.ExpressionContext,i)


        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterComparisonExpr" ):
                listener.enterComparisonExpr(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitComparisonExpr" ):
                listener.exitComparisonExpr(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitComparisonExpr" ):
                return visitor.visitComparisonExpr(self)
            else:
                return visitor.visitChildren(self)


    class LogicalNotExprContext(ExpressionContext):

        def __init__(self, parser, ctx:ParserRuleContext): # actually a ConstraintParser.ExpressionContext
            super().__init__(parser)
            self.copyFrom(ctx)

        def expression(self):
            return self.getTypedRuleContext(ConstraintParser.ExpressionContext,0)


        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterLogicalNotExpr" ):
                listener.enterLogicalNotExpr(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitLogicalNotExpr" ):
                listener.exitLogicalNotExpr(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitLogicalNotExpr" ):
                return visitor.visitLogicalNotExpr(self)
            else:
                return visitor.visitChildren(self)


    class ListAccessExprContext(ExpressionContext):

        def __init__(self, parser, ctx:ParserRuleContext): # actually a ConstraintParser.ExpressionContext
            super().__init__(parser)
            self.copyFrom(ctx)

        def expression(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(ConstraintParser.ExpressionContext)
            else:
                return self.getTypedRuleContext(ConstraintParser.ExpressionContext,i)


        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterListAccessExpr" ):
                listener.enterListAccessExpr(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitListAccessExpr" ):
                listener.exitListAccessExpr(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitListAccessExpr" ):
                return visitor.visitListAccessExpr(self)
            else:
                return visitor.visitChildren(self)


    class UnarySignExprContext(ExpressionContext):

        def __init__(self, parser, ctx:ParserRuleContext): # actually a ConstraintParser.ExpressionContext
            super().__init__(parser)
            self.copyFrom(ctx)

        def expression(self):
            return self.getTypedRuleContext(ConstraintParser.ExpressionContext,0)


        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterUnarySignExpr" ):
                listener.enterUnarySignExpr(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitUnarySignExpr" ):
                listener.exitUnarySignExpr(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitUnarySignExpr" ):
                return visitor.visitUnarySignExpr(self)
            else:
                return visitor.visitChildren(self)


    class LogicalAndExprContext(ExpressionContext):

        def __init__(self, parser, ctx:ParserRuleContext): # actually a ConstraintParser.ExpressionContext
            super().__init__(parser)
            self.copyFrom(ctx)

        def expression(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(ConstraintParser.ExpressionContext)
            else:
                return self.getTypedRuleContext(ConstraintParser.ExpressionContext,i)


        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterLogicalAndExpr" ):
                listener.enterLogicalAndExpr(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitLogicalAndExpr" ):
                listener.exitLogicalAndExpr(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitLogicalAndExpr" ):
                return visitor.visitLogicalAndExpr(self)
            else:
                return visitor.visitChildren(self)


    class PowerExprContext(ExpressionContext):

        def __init__(self, parser, ctx:ParserRuleContext): # actually a ConstraintParser.ExpressionContext
            super().__init__(parser)
            self.copyFrom(ctx)

        def expression(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(ConstraintParser.ExpressionContext)
            else:
                return self.getTypedRuleContext(ConstraintParser.ExpressionContext,i)


        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterPowerExpr" ):
                listener.enterPowerExpr(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitPowerExpr" ):
                listener.exitPowerExpr(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitPowerExpr" ):
                return visitor.visitPowerExpr(self)
            else:
                return visitor.visitChildren(self)


    class LogicalOrExprContext(ExpressionContext):

        def __init__(self, parser, ctx:ParserRuleContext): # actually a ConstraintParser.ExpressionContext
            super().__init__(parser)
            self.copyFrom(ctx)

        def expression(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(ConstraintParser.ExpressionContext)
            else:
                return self.getTypedRuleContext(ConstraintParser.ExpressionContext,i)


        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterLogicalOrExpr" ):
                listener.enterLogicalOrExpr(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitLogicalOrExpr" ):
                listener.exitLogicalOrExpr(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitLogicalOrExpr" ):
                return visitor.visitLogicalOrExpr(self)
            else:
                return visitor.visitChildren(self)


    class AdditionExprContext(ExpressionContext):

        def __init__(self, parser, ctx:ParserRuleContext): # actually a ConstraintParser.ExpressionContext
            super().__init__(parser)
            self.copyFrom(ctx)

        def expression(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(ConstraintParser.ExpressionContext)
            else:
                return self.getTypedRuleContext(ConstraintParser.ExpressionContext,i)


        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterAdditionExpr" ):
                listener.enterAdditionExpr(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitAdditionExpr" ):
                listener.exitAdditionExpr(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitAdditionExpr" ):
                return visitor.visitAdditionExpr(self)
            else:
                return visitor.visitChildren(self)


    class BitshiftExprContext(ExpressionContext):

        def __init__(self, parser, ctx:ParserRuleContext): # actually a ConstraintParser.ExpressionContext
            super().__init__(parser)
            self.copyFrom(ctx)

        def expression(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(ConstraintParser.ExpressionContext)
            else:
                return self.getTypedRuleContext(ConstraintParser.ExpressionContext,i)


        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterBitshiftExpr" ):
                listener.enterBitshiftExpr(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitBitshiftExpr" ):
                listener.exitBitshiftExpr(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitBitshiftExpr" ):
                return visitor.visitBitshiftExpr(self)
            else:
                return visitor.visitChildren(self)


    class BitwiseNotExprContext(ExpressionContext):

        def __init__(self, parser, ctx:ParserRuleContext): # actually a ConstraintParser.ExpressionContext
            super().__init__(parser)
            self.copyFrom(ctx)

        def expression(self):
            return self.getTypedRuleContext(ConstraintParser.ExpressionContext,0)


        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterBitwiseNotExpr" ):
                listener.enterBitwiseNotExpr(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitBitwiseNotExpr" ):
                listener.exitBitwiseNotExpr(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitBitwiseNotExpr" ):
                return visitor.visitBitwiseNotExpr(self)
            else:
                return visitor.visitChildren(self)


    class LiteralExprContext(ExpressionContext):

        def __init__(self, parser, ctx:ParserRuleContext): # actually a ConstraintParser.ExpressionContext
            super().__init__(parser)
            self.copyFrom(ctx)

        def integer(self):
            return self.getTypedRuleContext(ConstraintParser.IntegerContext,0)

        def float_(self):
            return self.getTypedRuleContext(ConstraintParser.FloatContext,0)

        def string(self):
            return self.getTypedRuleContext(ConstraintParser.StringContext,0)

        def boolean(self):
            return self.getTypedRuleContext(ConstraintParser.BooleanContext,0)


        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterLiteralExpr" ):
                listener.enterLiteralExpr(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitLiteralExpr" ):
                listener.exitLiteralExpr(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitLiteralExpr" ):
                return visitor.visitLiteralExpr(self)
            else:
                return visitor.visitChildren(self)


    class VariableExprContext(ExpressionContext):

        def __init__(self, parser, ctx:ParserRuleContext): # actually a ConstraintParser.ExpressionContext
            super().__init__(parser)
            self.copyFrom(ctx)

        def IDENTIFIER(self):
            return self.getToken(ConstraintParser.IDENTIFIER, 0)

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterVariableExpr" ):
                listener.enterVariableExpr(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitVariableExpr" ):
                listener.exitVariableExpr(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitVariableExpr" ):
                return visitor.visitVariableExpr(self)
            else:
                return visitor.visitChildren(self)


    class ListExprContext(ExpressionContext):

        def __init__(self, parser, ctx:ParserRuleContext): # actually a ConstraintParser.ExpressionContext
            super().__init__(parser)
            self.copyFrom(ctx)

        def expression(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(ConstraintParser.ExpressionContext)
            else:
                return self.getTypedRuleContext(ConstraintParser.ExpressionContext,i)


        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterListExpr" ):
                listener.enterListExpr(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitListExpr" ):
                listener.exitListExpr(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitListExpr" ):
                return visitor.visitListExpr(self)
            else:
                return visitor.visitChildren(self)


    class BitwiseXorExprContext(ExpressionContext):

        def __init__(self, parser, ctx:ParserRuleContext): # actually a ConstraintParser.ExpressionContext
            super().__init__(parser)
            self.copyFrom(ctx)

        def expression(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(ConstraintParser.ExpressionContext)
            else:
                return self.getTypedRuleContext(ConstraintParser.ExpressionContext,i)


        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterBitwiseXorExpr" ):
                listener.enterBitwiseXorExpr(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitBitwiseXorExpr" ):
                listener.exitBitwiseXorExpr(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitBitwiseXorExpr" ):
                return visitor.visitBitwiseXorExpr(self)
            else:
                return visitor.visitChildren(self)


    class ProductExprContext(ExpressionContext):

        def __init__(self, parser, ctx:ParserRuleContext): # actually a ConstraintParser.ExpressionContext
            super().__init__(parser)
            self.copyFrom(ctx)

        def expression(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(ConstraintParser.ExpressionContext)
            else:
                return self.getTypedRuleContext(ConstraintParser.ExpressionContext,i)


        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterProductExpr" ):
                listener.enterProductExpr(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitProductExpr" ):
                listener.exitProductExpr(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitProductExpr" ):
                return visitor.visitProductExpr(self)
            else:
                return visitor.visitChildren(self)



    def expression(self, _p:int=0):
        _parentctx = self._ctx
        _parentState = self.state
        localctx = ConstraintParser.ExpressionContext(self, self._ctx, _parentState)
        _prevctx = localctx
        _startState = 0
        self.enterRecursionRule(localctx, 0, self.RULE_expression, _p)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 53
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input,5,self._ctx)
            if la_ == 1:
                localctx = ConstraintParser.ParenthesisExprContext(self, localctx)
                self._ctx = localctx
                _prevctx = localctx

                self.state = 11
                self.match(ConstraintParser.T__0)
                self.state = 12
                self.expression(0)
                self.state = 13
                self.match(ConstraintParser.T__1)
                pass

            elif la_ == 2:
                localctx = ConstraintParser.BitwiseNotExprContext(self, localctx)
                self._ctx = localctx
                _prevctx = localctx
                self.state = 15
                self.match(ConstraintParser.T__5)
                self.state = 16
                self.expression(16)
                pass

            elif la_ == 3:
                localctx = ConstraintParser.UnarySignExprContext(self, localctx)
                self._ctx = localctx
                _prevctx = localctx
                self.state = 17
                _la = self._input.LA(1)
                if not(_la==7 or _la==8):
                    self._errHandler.recoverInline(self)
                else:
                    self._errHandler.reportMatch(self)
                    self.consume()
                self.state = 18
                self.expression(15)
                pass

            elif la_ == 4:
                localctx = ConstraintParser.LogicalNotExprContext(self, localctx)
                self._ctx = localctx
                _prevctx = localctx
                self.state = 19
                self.match(ConstraintParser.T__22)
                self.state = 20
                self.expression(7)
                pass

            elif la_ == 5:
                localctx = ConstraintParser.VariableExprContext(self, localctx)
                self._ctx = localctx
                _prevctx = localctx
                self.state = 21
                self.match(ConstraintParser.IDENTIFIER)
                pass

            elif la_ == 6:
                localctx = ConstraintParser.FunctionExprContext(self, localctx)
                self._ctx = localctx
                _prevctx = localctx
                self.state = 22
                self.match(ConstraintParser.IDENTIFIER)
                self.state = 23
                self.match(ConstraintParser.T__0)
                self.state = 32
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if (((_la) & ~0x3f) == 0 and ((1 << _la) & 25241321930) != 0):
                    self.state = 24
                    self.expression(0)
                    self.state = 29
                    self._errHandler.sync(self)
                    _la = self._input.LA(1)
                    while _la==28:
                        self.state = 25
                        self.match(ConstraintParser.T__27)
                        self.state = 26
                        self.expression(0)
                        self.state = 31
                        self._errHandler.sync(self)
                        _la = self._input.LA(1)



                self.state = 34
                self.match(ConstraintParser.T__1)
                pass

            elif la_ == 7:
                localctx = ConstraintParser.ListExprContext(self, localctx)
                self._ctx = localctx
                _prevctx = localctx
                self.state = 35
                self.match(ConstraintParser.T__2)
                self.state = 44
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if (((_la) & ~0x3f) == 0 and ((1 << _la) & 25241321930) != 0):
                    self.state = 36
                    self.expression(0)
                    self.state = 41
                    self._errHandler.sync(self)
                    _la = self._input.LA(1)
                    while _la==28:
                        self.state = 37
                        self.match(ConstraintParser.T__27)
                        self.state = 38
                        self.expression(0)
                        self.state = 43
                        self._errHandler.sync(self)
                        _la = self._input.LA(1)



                self.state = 46
                self.match(ConstraintParser.T__3)
                pass

            elif la_ == 8:
                localctx = ConstraintParser.LiteralExprContext(self, localctx)
                self._ctx = localctx
                _prevctx = localctx
                self.state = 51
                self._errHandler.sync(self)
                token = self._input.LA(1)
                if token in [29]:
                    self.state = 47
                    self.integer()
                    pass
                elif token in [31]:
                    self.state = 48
                    self.float_()
                    pass
                elif token in [30]:
                    self.state = 49
                    self.string()
                    pass
                elif token in [32]:
                    self.state = 50
                    self.boolean()
                    pass
                else:
                    raise NoViableAltException(self)

                pass


            self._ctx.stop = self._input.LT(-1)
            self.state = 92
            self._errHandler.sync(self)
            _alt = self._interp.adaptivePredict(self._input,7,self._ctx)
            while _alt!=2 and _alt!=ATN.INVALID_ALT_NUMBER:
                if _alt==1:
                    if self._parseListeners is not None:
                        self.triggerExitRuleEvent()
                    _prevctx = localctx
                    self.state = 90
                    self._errHandler.sync(self)
                    la_ = self._interp.adaptivePredict(self._input,6,self._ctx)
                    if la_ == 1:
                        localctx = ConstraintParser.PowerExprContext(self, ConstraintParser.ExpressionContext(self, _parentctx, _parentState))
                        self.pushNewRecursionContext(localctx, _startState, self.RULE_expression)
                        self.state = 55
                        if not self.precpred(self._ctx, 17):
                            from antlr4.error.Errors import FailedPredicateException
                            raise FailedPredicateException(self, "self.precpred(self._ctx, 17)")
                        self.state = 56
                        self.match(ConstraintParser.T__4)
                        self.state = 57
                        self.expression(18)
                        pass

                    elif la_ == 2:
                        localctx = ConstraintParser.ProductExprContext(self, ConstraintParser.ExpressionContext(self, _parentctx, _parentState))
                        self.pushNewRecursionContext(localctx, _startState, self.RULE_expression)
                        self.state = 58
                        if not self.precpred(self._ctx, 14):
                            from antlr4.error.Errors import FailedPredicateException
                            raise FailedPredicateException(self, "self.precpred(self._ctx, 14)")
                        self.state = 59
                        _la = self._input.LA(1)
                        if not((((_la) & ~0x3f) == 0 and ((1 << _la) & 3584) != 0)):
                            self._errHandler.recoverInline(self)
                        else:
                            self._errHandler.reportMatch(self)
                            self.consume()
                        self.state = 60
                        self.expression(15)
                        pass

                    elif la_ == 3:
                        localctx = ConstraintParser.AdditionExprContext(self, ConstraintParser.ExpressionContext(self, _parentctx, _parentState))
                        self.pushNewRecursionContext(localctx, _startState, self.RULE_expression)
                        self.state = 61
                        if not self.precpred(self._ctx, 13):
                            from antlr4.error.Errors import FailedPredicateException
                            raise FailedPredicateException(self, "self.precpred(self._ctx, 13)")
                        self.state = 62
                        _la = self._input.LA(1)
                        if not(_la==7 or _la==8):
                            self._errHandler.recoverInline(self)
                        else:
                            self._errHandler.reportMatch(self)
                            self.consume()
                        self.state = 63
                        self.expression(14)
                        pass

                    elif la_ == 4:
                        localctx = ConstraintParser.BitshiftExprContext(self, ConstraintParser.ExpressionContext(self, _parentctx, _parentState))
                        self.pushNewRecursionContext(localctx, _startState, self.RULE_expression)
                        self.state = 64
                        if not self.precpred(self._ctx, 12):
                            from antlr4.error.Errors import FailedPredicateException
                            raise FailedPredicateException(self, "self.precpred(self._ctx, 12)")
                        self.state = 65
                        _la = self._input.LA(1)
                        if not(_la==12 or _la==13):
                            self._errHandler.recoverInline(self)
                        else:
                            self._errHandler.reportMatch(self)
                            self.consume()
                        self.state = 66
                        self.expression(13)
                        pass

                    elif la_ == 5:
                        localctx = ConstraintParser.BitwiseAndExprContext(self, ConstraintParser.ExpressionContext(self, _parentctx, _parentState))
                        self.pushNewRecursionContext(localctx, _startState, self.RULE_expression)
                        self.state = 67
                        if not self.precpred(self._ctx, 11):
                            from antlr4.error.Errors import FailedPredicateException
                            raise FailedPredicateException(self, "self.precpred(self._ctx, 11)")
                        self.state = 68
                        self.match(ConstraintParser.T__13)
                        self.state = 69
                        self.expression(12)
                        pass

                    elif la_ == 6:
                        localctx = ConstraintParser.BitwiseXorExprContext(self, ConstraintParser.ExpressionContext(self, _parentctx, _parentState))
                        self.pushNewRecursionContext(localctx, _startState, self.RULE_expression)
                        self.state = 70
                        if not self.precpred(self._ctx, 10):
                            from antlr4.error.Errors import FailedPredicateException
                            raise FailedPredicateException(self, "self.precpred(self._ctx, 10)")
                        self.state = 71
                        self.match(ConstraintParser.T__14)
                        self.state = 72
                        self.expression(11)
                        pass

                    elif la_ == 7:
                        localctx = ConstraintParser.BitwiseOrExprContext(self, ConstraintParser.ExpressionContext(self, _parentctx, _parentState))
                        self.pushNewRecursionContext(localctx, _startState, self.RULE_expression)
                        self.state = 73
                        if not self.precpred(self._ctx, 9):
                            from antlr4.error.Errors import FailedPredicateException
                            raise FailedPredicateException(self, "self.precpred(self._ctx, 9)")
                        self.state = 74
                        self.match(ConstraintParser.T__15)
                        self.state = 75
                        self.expression(10)
                        pass

                    elif la_ == 8:
                        localctx = ConstraintParser.ComparisonExprContext(self, ConstraintParser.ExpressionContext(self, _parentctx, _parentState))
                        self.pushNewRecursionContext(localctx, _startState, self.RULE_expression)
                        self.state = 76
                        if not self.precpred(self._ctx, 8):
                            from antlr4.error.Errors import FailedPredicateException
                            raise FailedPredicateException(self, "self.precpred(self._ctx, 8)")
                        self.state = 77
                        _la = self._input.LA(1)
                        if not((((_la) & ~0x3f) == 0 and ((1 << _la) & 8257536) != 0)):
                            self._errHandler.recoverInline(self)
                        else:
                            self._errHandler.reportMatch(self)
                            self.consume()
                        self.state = 78
                        self.expression(9)
                        pass

                    elif la_ == 9:
                        localctx = ConstraintParser.LogicalAndExprContext(self, ConstraintParser.ExpressionContext(self, _parentctx, _parentState))
                        self.pushNewRecursionContext(localctx, _startState, self.RULE_expression)
                        self.state = 79
                        if not self.precpred(self._ctx, 6):
                            from antlr4.error.Errors import FailedPredicateException
                            raise FailedPredicateException(self, "self.precpred(self._ctx, 6)")
                        self.state = 80
                        _la = self._input.LA(1)
                        if not(_la==24 or _la==25):
                            self._errHandler.recoverInline(self)
                        else:
                            self._errHandler.reportMatch(self)
                            self.consume()
                        self.state = 81
                        self.expression(7)
                        pass

                    elif la_ == 10:
                        localctx = ConstraintParser.LogicalOrExprContext(self, ConstraintParser.ExpressionContext(self, _parentctx, _parentState))
                        self.pushNewRecursionContext(localctx, _startState, self.RULE_expression)
                        self.state = 82
                        if not self.precpred(self._ctx, 5):
                            from antlr4.error.Errors import FailedPredicateException
                            raise FailedPredicateException(self, "self.precpred(self._ctx, 5)")
                        self.state = 83
                        _la = self._input.LA(1)
                        if not(_la==26 or _la==27):
                            self._errHandler.recoverInline(self)
                        else:
                            self._errHandler.reportMatch(self)
                            self.consume()
                        self.state = 84
                        self.expression(6)
                        pass

                    elif la_ == 11:
                        localctx = ConstraintParser.ListAccessExprContext(self, ConstraintParser.ExpressionContext(self, _parentctx, _parentState))
                        self.pushNewRecursionContext(localctx, _startState, self.RULE_expression)
                        self.state = 85
                        if not self.precpred(self._ctx, 18):
                            from antlr4.error.Errors import FailedPredicateException
                            raise FailedPredicateException(self, "self.precpred(self._ctx, 18)")
                        self.state = 86
                        self.match(ConstraintParser.T__2)
                        self.state = 87
                        self.expression(0)
                        self.state = 88
                        self.match(ConstraintParser.T__3)
                        pass

             
                self.state = 94
                self._errHandler.sync(self)
                _alt = self._interp.adaptivePredict(self._input,7,self._ctx)

        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.unrollRecursionContexts(_parentctx)
        return localctx


    class IntegerContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def INTEGER(self):
            return self.getToken(ConstraintParser.INTEGER, 0)

        def getRuleIndex(self):
            return ConstraintParser.RULE_integer

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterInteger" ):
                listener.enterInteger(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitInteger" ):
                listener.exitInteger(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitInteger" ):
                return visitor.visitInteger(self)
            else:
                return visitor.visitChildren(self)




    def integer(self):

        localctx = ConstraintParser.IntegerContext(self, self._ctx, self.state)
        self.enterRule(localctx, 2, self.RULE_integer)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 95
            self.match(ConstraintParser.INTEGER)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class FloatContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def FLOAT(self):
            return self.getToken(ConstraintParser.FLOAT, 0)

        def getRuleIndex(self):
            return ConstraintParser.RULE_float

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterFloat" ):
                listener.enterFloat(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitFloat" ):
                listener.exitFloat(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitFloat" ):
                return visitor.visitFloat(self)
            else:
                return visitor.visitChildren(self)




    def float_(self):

        localctx = ConstraintParser.FloatContext(self, self._ctx, self.state)
        self.enterRule(localctx, 4, self.RULE_float)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 97
            self.match(ConstraintParser.FLOAT)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class StringContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def STRING(self):
            return self.getToken(ConstraintParser.STRING, 0)

        def getRuleIndex(self):
            return ConstraintParser.RULE_string

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterString" ):
                listener.enterString(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitString" ):
                listener.exitString(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitString" ):
                return visitor.visitString(self)
            else:
                return visitor.visitChildren(self)




    def string(self):

        localctx = ConstraintParser.StringContext(self, self._ctx, self.state)
        self.enterRule(localctx, 6, self.RULE_string)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 99
            self.match(ConstraintParser.STRING)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class BooleanContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def BOOLEAN(self):
            return self.getToken(ConstraintParser.BOOLEAN, 0)

        def getRuleIndex(self):
            return ConstraintParser.RULE_boolean

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterBoolean" ):
                listener.enterBoolean(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitBoolean" ):
                listener.exitBoolean(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitBoolean" ):
                return visitor.visitBoolean(self)
            else:
                return visitor.visitChildren(self)




    def boolean(self):

        localctx = ConstraintParser.BooleanContext(self, self._ctx, self.state)
        self.enterRule(localctx, 8, self.RULE_boolean)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 101
            self.match(ConstraintParser.BOOLEAN)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx



    def sempred(self, localctx:RuleContext, ruleIndex:int, predIndex:int):
        if self._predicates == None:
            self._predicates = dict()
        self._predicates[0] = self.expression_sempred
        pred = self._predicates.get(ruleIndex, None)
        if pred is None:
            raise Exception("No predicate with index:" + str(ruleIndex))
        else:
            return pred(localctx, predIndex)

    def expression_sempred(self, localctx:ExpressionContext, predIndex:int):
            if predIndex == 0:
                return self.precpred(self._ctx, 17)
         

            if predIndex == 1:
                return self.precpred(self._ctx, 14)
         

            if predIndex == 2:
                return self.precpred(self._ctx, 13)
         

            if predIndex == 3:
                return self.precpred(self._ctx, 12)
         

            if predIndex == 4:
                return self.precpred(self._ctx, 11)
         

            if predIndex == 5:
                return self.precpred(self._ctx, 10)
         

            if predIndex == 6:
                return self.precpred(self._ctx, 9)
         

            if predIndex == 7:
                return self.precpred(self._ctx, 8)
         

            if predIndex == 8:
                return self.precpred(self._ctx, 6)
         

            if predIndex == 9:
                return self.precpred(self._ctx, 5)
         

            if predIndex == 10:
                return self.precpred(self._ctx, 18)
         




