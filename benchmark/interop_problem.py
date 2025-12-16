from collections.abc import Callable
import math

from interopt import Study
from interopt.parameter import (
    Boolean as BooleanInteropt,
    Categorical as CategoricalInteropt,
    Integer as IntegerInteropt,
    IntExponential as IntExponentialInteropt,
    Permutation as PermutationInteropt,
    Real as RealInteropt,
)

from schedgehammer.problem import Problem
from schedgehammer.param_types import Param
from schedgehammer.param_types import (
    CategoricalParam,
    ExpIntParam,
    IntegerParam,
    PermutationParam,
    RealParam,
    SwitchParam,
)


class CatbenchProblem(Problem):
    def __init__(
        self,
        name: str,
        params: dict[str, Param],
        study: Study,
        constraints: list[str] = [],
    ):
        super().__init__(name, params, constraints, init_solver=True)
        self.fidelity_params = {}
        for fidelity_param in study.definition.search_space.fidelity_params:
            self.fidelity_params[fidelity_param.name] = fidelity_param.default
        self.study = study

    def cost_function(self, config: dict[str, any]) -> float:
        config = config.copy()
        for name, val in config.items():
            if type(val) is list:
                config[name] = str(val)

        # print(f"query: {config}")
        # print(f"fidelyti param: {self.fidelity_params}")
        # print("query")
        result = self.study.query(config, self.fidelity_params)["compute_time"]
        # print(f"query result: {result}")

        return result


def problem_from_study(study: Study) -> Problem:
    params = {}
    for param in study.definition.search_space.params:
        if type(param) is CategoricalInteropt:
            params[param.name] = CategoricalParam(
                values=param.categories,
            )
        elif type(param) is PermutationInteropt:
            params[param.name] = PermutationParam(values=list(param.default))
        elif type(param) is BooleanInteropt:
            params[param.name] = SwitchParam()
        elif type(param) is IntegerInteropt:
            params[param.name] = IntegerParam(
                min_val=param.bounds[0],
                max_val=param.bounds[1],
            )
        elif type(param) is RealInteropt:
            params[param.name] = RealParam(
                min_val=param.bounds[0],
                max_val=param.bounds[1],
            )
        elif type(param) is IntExponentialInteropt:
            params[param.name] = ExpIntParam(
                base=param.base,
                min_exp=math.log(param.bounds[0], param.base),
                max_exp=math.log(param.bounds[1], param.base),
            )
        else:
            raise ValueError(f"Problem got unsupported parameter type: {type(param)}")

    # def interop_eval(config):
    #     config = config.copy()
    #     for name, val in config.items():
    #         if type(val) is list:
    #             config[name] = str(val)
    #     return study.query(config, fidelity_params)["compute_time"]

    return CatbenchProblem(
        study.definition.name,
        params,
        study,
        [c.constraint for c in study.definition.search_space.constraints],
    )
