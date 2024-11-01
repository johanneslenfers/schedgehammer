from schedgehammer.cost import cost
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

    budget = EvalBudget(10000)
    # budget = TimeBudget(2.5)
    config, score = GeneticTuner(problem, budget).tune()


if __name__ == "__main__":
    main()
