grammar Constraint;

expression: '(' expression ')'  # ParenthesisExpr
    | expression '[' expression ']'  # ListAccessExpr
    | expression '**' expression  # PowerExpr
    | '~' expression  # BitwiseNotExpr
    | ('+' | '-') expression  # UnarySignExpr
    | expression ('*' | '/' | '%') expression  # ProductExpr
    | expression ('+' | '-') expression  # AdditionExpr
    | expression ('<<' | '>>' ) expression  # BitshiftExpr
    | expression '&' expression  # BitwiseAndExpr
    | expression '^' expression  # BitwiseXorExpr
    | expression '|' expression  # BitwiseOrExpr
    | expression ('==' | '!=' | '>' | '>=' | '<' | '<=') expression  # ComparisonExpr
    | 'not' expression  # LogicalNotExpr
    | expression ('and' | '&&') expression  # LogicalAndExpr
    | expression ('or' | '||') expression  # LogicalOrExpr
    | IDENTIFIER  # VariableExpr
    | IDENTIFIER '(' (expression (',' expression)* )? ')'  # FunctionExpr
    | '[' (expression (',' expression)* )? ']'  # ListExpr
    | (integer | float | string | boolean)  # LiteralExpr
    ;

integer: INTEGER;
float: FLOAT;
string: STRING;
boolean: BOOLEAN;

INTEGER: DIGIT+;
STRING: '\'' CHARACTER* '\'' | '"' CHARACTER* '"';
FLOAT: ((DIGIT+ '.' DIGIT*) | ('.' DIGIT+)) ('e' SIGN? DIGIT+)?;
BOOLEAN: 'True' | 'False';
SIGN: '+' | '-';
IDENTIFIER: LETTER (LETTER | DIGIT | '_')*;
fragment CHARACTER: LETTER | DIGIT | '_';
fragment DIGIT: [0-9];
fragment LETTER: [a-zA-Z];

IGNORE : (' ' | '\t' | '\n' | '\r' | EOF ) -> skip ;
