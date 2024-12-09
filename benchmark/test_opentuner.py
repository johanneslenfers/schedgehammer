# Only needed since this is in the same repo as schedgehammer.
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
##############################################################

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
import catbench as cb
import opentuner
from opentuner.resultsdb.models import Input
from opentuner.search.manipulator import BooleanParameter, PowerOfTwoParameter
from opentuner import ConfigurationManipulator, EnumParameter, PermutationParameter, FloatParameter
from opentuner import IntegerParameter
from opentuner import MeasurementInterface
from opentuner import Result
from schedgehammer.constraint import Constraint

def translate_param(param):
    if type(param) is CategoricalInteropt:
        return EnumParameter(param.name, param.categories)
    elif type(param) is PermutationInteropt:
        return PermutationParameter(param.name, list(param.default))
    elif type(param) is BooleanInteropt:
        return BooleanParameter(param.name)
    elif type(param) is IntegerInteropt:
        return IntegerParameter(param.name, param.bounds[0], param.bounds[1])
    elif type(param) is RealInteropt:
        return FloatParameter(param.name, param.bounds[0], param.bounds[1])
    elif type(param) is IntExponentialInteropt:
        assert param.base == 2
        return PowerOfTwoParameter(param.name, param.bounds[0], param.bounds[1])
    else:
        raise ValueError(f"Problem got unsupported parameter type: {type(param)}")


class OpentunerCatbenchAdapter(MeasurementInterface):
    study: Study
    constraints: list[Constraint]

    def __init__(self, study: Study, args):
        self.study = study
        self.constraints = [Constraint(s.constraint) for s in study.definition.search_space.constraints]
        super().__init__(args)

    def manipulator(self):
        """
        Define the search space by creating a
        ConfigurationManipulator
        """
        manipulator = ConfigurationManipulator()
        for param in self.study.definition.search_space.params:
            manipulator.add_parameter(translate_param(param))
        return manipulator

    def run(self, desired_result, input: Input, limit):
        """
        Compile and run a given configuration then
        return performance
        """
        config = desired_result.configuration.data.copy()

        fidelity_params = {}
        for fidelity_param in study.definition.search_space.fidelity_params:
            fidelity_params[fidelity_param.name] = fidelity_param.default

        for name, val in config.items():
            if type(val) is list:
                config[name] = str(val)

        for constraint in self.constraints:
            if not constraint.evaluate(config):
                return Result(time=math.inf)

        return Result(time=self.study.query(config, fidelity_params)["compute_time"])

ITERATIONS = 500
BENCHMARKS = [
    "spmm",
    "spmv",
    "sddmm",
    "mttkrp",
    "ttv",
    "asum",
    "harris",
    "kmeans",
    "stencil",
]

if __name__ == "__main__":

    argparser = opentuner.default_argparser()

    for benchmark_name in BENCHMARKS:
        study = cb.benchmark(benchmark_name)
        from opentuner.tuningrunmain import TuningRunMain
        for i in range(10):
            args = argparser.parse_args(['--test-limit', str(ITERATIONS), '--label', benchmark_name])
            TuningRunMain(OpentunerCatbenchAdapter(study, args), args).main()

