# Only needed since this is in the same repo as schedgehammer.
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
#################################################################

import numpy as np

from schedgehammer.genetic_tuner import GeneticTuner
from schedgehammer.param_types import (
    SwitchParam,
    RealParam,
    IntegerParam,
    OrdinalParam,
    CategoricalParam,
    PermutationParam,
)
from schedgehammer.problem import Problem
from schedgehammer.tuner import EvalBudget, TimeBudget


def main():
    problem = Problem(
        {
            "magic": SwitchParam(),
            "mana": RealParam(0, 10),
            "level": IntegerParam(1, 100),
            "power": OrdinalParam([1, 2, 4, 8, 16]),
            "creature": CategoricalParam(
                [
                    "dwarf",
                    "halfling",
                    "gold_golem",
                    "mage",
                    "naga",
                    "genie",
                    "dragon_golem",
                    "titan",
                ]
            ),
            "order": PermutationParam([1, 2, 3, 4, 5]),
        },
        cost_function=lambda x: -cost(x),
    )  # Make minimization problem.

    be = EvalBudget(10000)
    bt = TimeBudget(1.5)
    result = GeneticTuner(problem, [be, bt]).tune()
    result.generate_csv("results/evaluations.csv")
    result.generate_plot("results/plot.png")

# a synthetic cost function
def cost(configuration: dict[str, bool | float | int | str | list[int]]) -> float:
    # parse value of tuning parameters from configuration
    switch: bool = configuration["magic"]  # type: ignore
    real_param: float = configuration["mana"]  # type: ignore
    integer_param: int = configuration["level"]  # type: ignore
    ordinal_param: int = configuration["power"]  # type: ignore
    categorical_param: str = configuration["creature"]  # type: ignore
    permutation_param: list[int] = configuration["order"]  # type: ignore

    # Switch/Boolean contribution with a jump behavior
    switch_contrib: float = 0.1 if switch else 2.0  # Much larger difference between True and False

    # Real parameter
    term1: float = float(real_param * np.cos(np.sin(abs(real_param ** 2))) ** 2 - 0.5)  # type: ignore
    term2: float = (1 + 0.001 * (real_param ** 2)) ** 2  # type: ignore
    real_contrib: float = 0.5 + term1 / term2

    # Integer parameter: introducing non-linear jumps based on specific thresholds
    if integer_param <= 1:
        integer_contrib = 0.9
    elif 1 < integer_param <= 5:
        integer_contrib = 0.5
    else:
        integer_contrib: float = -0.5

        # Ordinal parameter with stepwise jumps
    if ordinal_param == 1:
        ordinal_contrib = 1.0
    elif ordinal_param == 2:
        ordinal_contrib = 0.7
    elif ordinal_param == 4:
        ordinal_contrib = 0.2
    elif ordinal_param == 8:
        ordinal_contrib = 0.65
    else:
        ordinal_contrib: float = -0.5

        # Categorical contribution with fixed scores
    categorical_mapping: dict[str, float] = {
        'dwarf': -0.7,
        'halfling': -0.2,
        'gold_golem': 0.0,
        'mage': 1,
        'naga': 1.2,
        'genie': 1.5,
        'dragon_golem': 2.5,
        'titan': 5,
    }
    categorical_contrib: float = categorical_mapping.get(categorical_param, 0.0)

    # If we have a mage or a genie and magic is activaed we add a bonus
    if switch:
        if categorical_param == 'mage' or categorical_param == 'genie':
            switch_contrib = 5.0

    # If we have a non-magic creature and mana is high activated we add a penalty
    if real_param > 2.0:
        if categorical_param != 'mage' and categorical_param != 'genie':
            real_contrib = -1.0

    # Permutation contribution with more non-linear effects
    permutation_contrib: int = sum(
        abs(permutation_param[i] - permutation_param[i + 1]) for i in range(len(permutation_param) - 1))
    if permutation_contrib < 3:
        perm_contrib = 1.0
    elif permutation_contrib < 5:
        perm_contrib = 0.2
    else:
        perm_contrib: float = -1.0

        # Combine contributions to get the performance value
    performance_value: float = (switch_contrib +
                                real_contrib +
                                integer_contrib +
                                ordinal_contrib +
                                categorical_contrib +
                                perm_contrib)

    return performance_value



if __name__ == "__main__":
    main()
