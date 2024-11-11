from schedgehammer.param_types import ParamValue

class Constraint:
    dependencies: set[str]
    constraint_string: str

    def __init__(self, constraint: str, dependencies: list[str]):
        self.constraint_string = constraint
        self.dependencies = dependencies
        for dep in self.dependencies:
            self.constraint_string = self.constraint_string.replace(dep, f"config[\"{dep}\"]")

    def evaluate(self, config: dict[str, ParamValue]):
        return eval(self.constraint_string)
        
