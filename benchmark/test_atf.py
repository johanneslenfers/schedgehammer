# Only needed since this is in the same repo as schedgehammer.
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
##############################################################

import itertools
from dataclasses import dataclass
from typing import Callable

from pyatf import Tuner, Interval, Set, TP
from pyatf.abort_conditions import Evaluations
from pyatf.tuning_data import Configuration, Cost

from schedgehammer.constraint import Constraint

import math
from interopt.parameter import (
    Boolean as BooleanInteropt,
    Categorical as CategoricalInteropt,
    Integer as IntegerInteropt,
    IntExponential as IntExponentialInteropt,
    Permutation as PermutationInteropt,
    Real as RealInteropt, Param,
)
import catbench as cb

def translate_param(param):
    if type(param) is CategoricalInteropt:
        return Set(*param.categories)
    elif type(param) is PermutationInteropt:
        return Set(*[list(p) for p in itertools.permutations(list(param.default))])
    elif type(param) is BooleanInteropt:
        return Set(True, False)
    elif type(param) is IntegerInteropt:
        return Interval(param.bounds[0], param.bounds[1])
    elif type(param) is RealInteropt:
        return Interval(param.bounds[0], param.bounds[1], (param.bounds[1] - param.bounds[0]) / 1000)
    elif type(param) is IntExponentialInteropt:
        assert param.base == 2
        min_power = int(math.log(param.bounds[0], param.base))
        max_power = int(math.log(param.bounds[1], param.base))
        return Interval(min_power, max_power)
    else:
        raise ValueError(f"Problem got unsupported parameter type: {type(param)}")

@dataclass
class MoreGenericCostFunction:
    fun: Callable

    def __call__(self, configuration: Configuration) -> Cost:
        return self.fun(configuration)

ITERATIONS = 1000
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

def calculate_generation_order(params: list[str], constraints: list[Constraint]):
    generation_order = []
    used_params = []
    remaining_params = set(params)
    remaining_constraints = set(constraints)

    while len(remaining_params) > 0:
        best_param = None
        best_resolved_constraints = []

        for param in remaining_params:
            resolved_constraints = []
            for constraint in remaining_constraints:
                if constraint.dependencies.issubset(set(used_params + [param])):
                    resolved_constraints.append(constraint)
            if len(resolved_constraints) > len(best_resolved_constraints):
                best_resolved_constraints = resolved_constraints
                best_param = param

        if best_param is None:
            best_param = list(remaining_params)[0]

        generation_order.append((best_param, best_resolved_constraints))
        remaining_params.remove(best_param)
        used_params.append(best_param)
        remaining_constraints.difference_update(best_resolved_constraints)
    return generation_order

def create_checking_function(constraints: list[Constraint], params: dict[str, Param]):
    union = set()
    for constraint in constraints:
        union |= constraint.dependencies

    config_str = ", ".join([f"\"{name}\": {'2 ** ' if isinstance(params[name], IntExponentialInteropt) else ''}{name}" for name in union])
    if len(union) == 0:
        return None

    return eval(f"lambda {', '.join(union)}: all([constraint.evaluate({{{config_str}}}) for constraint in constraints])",
                {"constraints": constraints})

def main():
    for benchmark_name in BENCHMARKS:
        for i in range(50):
            study = cb.benchmark(benchmark_name)

            params = {p.name: p for p in study.definition.search_space.params}

            print(f"{benchmark_name}: {i}")

            constraints = [Constraint(constraint.constraint) for constraint in study.definition.search_space.constraints]
            generation_order = calculate_generation_order(list(params.keys()), constraints)
            tps = [TP(name, translate_param(params[name]), create_checking_function(constraints, params)) for name, constraints in generation_order]

            def cost_function(config):
                config = config.copy()

                fidelity_params = {}
                for fidelity_param in study.definition.search_space.fidelity_params:
                    fidelity_params[fidelity_param.name] = fidelity_param.default

                for name, val in config.items():
                    if type(val) is list:
                        config[name] = str(val)
                    if isinstance(params[name], IntExponentialInteropt):
                        config[name] = params[name].base ** config[name]

                return study.query(config, fidelity_params)["compute_time"]

            tuner = (Tuner()
                     .tuning_parameters(*tps)
                     .log_file(f'results/atf/{benchmark_name}/json/{i}.json')
                     .verbosity(0))
            tuner.tune(cost_function, Evaluations(ITERATIONS))


if __name__ == "__main__":
    main()
