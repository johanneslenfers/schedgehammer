from dataclasses import dataclass
from math import log
from typing import Callable

from interopt import Study
from interopt.parameter import (
    Boolean as BooleanInteropt,
)
from interopt.parameter import (
    Categorical as CategoricalInteropt,
)
from interopt.parameter import (
    Integer as IntegerInteropt,
)
from interopt.parameter import (
    IntExponential as IntExponentialInteropt,
)
from interopt.parameter import (
    Ordinal as OrdinalInteropt,
)
from interopt.parameter import (
    Permutation as PermutationInteropt,
)
from interopt.parameter import (
    Real as RealInteropt,
)

from schedgehammer.param_types import (
    CategoricalParam,
    ExpIntParam,
    IntegerParam,
    OrdinalParam,
    Param,
    ParamValue,
    PermutationParam,
    RealParam,
    SwitchParam,
)


@dataclass
class Problem:
    params: dict[str, Param]
    cost_function: Callable[[dict[str, ParamValue]], float] | None = None
    study: Study | None = None
    fidelity_params: dict[str, ParamValue] | None = None

    @classmethod
    def from_interopt_format(cls, study: Study) -> "Problem":
        params = {}
        for param in study.definition.search_space.params:
            if type(param) is CategoricalInteropt:
                params[param.name] = CategoricalParam(
                    values=param.categories,
                )
            elif type(param) is PermutationInteropt:
                params[param.name] = PermutationParam(values=param.default)
            elif type(param) is BooleanInteropt:
                params[param.name] = SwitchParam()
            elif type(param) is IntegerInteropt:
                params[param.name] = IntegerParam(
                    min_val=param.bounds[0],
                    max_val=param.bounds[1],
                )
            elif type(param) is OrdinalInteropt:
                params[param.name] = OrdinalParam(
                    values=[v for v in range(param.bounds[0], param.bounds[1] + 1)],
                )
            elif type(param) is RealInteropt:
                params[param.name] = RealParam(
                    min_val=param.bounds[0],
                    max_val=param.bounds[1],
                )
            elif type(param) is IntExponentialInteropt:
                params[param.name] = ExpIntParam(
                    base=param.base,
                    min_exp=log(param.bounds[0], param.base),
                    max_exp=log(param.bounds[1], param.base),
                )
            else:
                raise ValueError(
                    f"Problem got unsupported parameter type: {type(param)}"
                )
        fidelity_params = {}
        for fidelity_param in study.definition.search_space.fidelity_params:
            fidelity_params[fidelity_param.name] = fidelity_param.default
        return cls(params, study=study, fidelity_params=fidelity_params)
