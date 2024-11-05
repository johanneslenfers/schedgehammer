from cost import cost
from random_search import RandomSearch
from local_search import LocalSearch

from tuning_problem import TuningProblem, BooleanParameter, RealParameter, IntegerParameter, \
    OrdinalParameter, CategoricalParameter, PermutationParameter

my_tuning_problem = TuningProblem({
            "magic": BooleanParameter(),
            "mana": RealParameter(0, 10),
            "level": IntegerParameter(1, 100),
            "power": OrdinalParameter([1, 2, 4, 8, 16]),
            "creature": CategoricalParameter(['dwarf', 'halfling', 'gold_golem', 'mage', 'naga', 'genie', 'dragon_golem', 'titan']),
            "order": PermutationParameter(5),
        }, lambda x: -cost(x)  # Invert function to make it minimization problem.
)

# RandomSearch(my_tuning_problem).forever()
LocalSearch(my_tuning_problem).forever()
