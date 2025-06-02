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
from schedgehammer.param_types import (
    CategoricalParam,
    ExpIntParam,
    IntegerParam,
    PermutationParam,
    RealParam,
    SwitchParam, ParamValue,
)

class ProblemFromStudy(Problem):

    study: Study

    def __init__(self, study):
        self.study = study
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

        super().__init__(
            study.definition.name,
            params,
            [c.constraint for c in study.definition.search_space.constraints],
            False
        )

    def cost_function(self, config: dict[str, ParamValue]) -> float:
        fidelity_params = {}
        for fidelity_param in self.study.definition.search_space.fidelity_params:
            fidelity_params[fidelity_param.name] = fidelity_param.default

        config = config.copy()
        for name, val in config.items():
            if type(val) is list:
                config[name] = str(val)
        return self.study.query(config, fidelity_params)["compute_time"]


def problem_from_study(study: Study) -> Problem:
    return ProblemFromStudy(study)
