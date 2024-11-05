import math

from tuner import Tuner
from tuning_problem import BooleanParameter, IntegerParameter, RealParameter, \
    OrdinalParameter, CategoricalParameter, PermutationParameter

class LocalSearch(Tuner):
    def generate_tuning(self):
        best_solution = None
        best_score = math.inf

        # Begin with a few random configs; take the best.
        for i in range(10):
            solution = self.random_config()
            score = self.tuning_problem.cost(solution)
            yield solution, score
            if score < best_score:
                best_solution = solution
                best_score = score

        current_solution = best_solution
        current_score = best_score
        while True:
            change = False
            # Go through all parameters, find neighbors for each parameter and test if it performs better.
            for name in self.tuning_problem.parameters:
                current_value = current_solution[name]
                neighbors = []
                match self.tuning_problem.parameters[name]:
                    case BooleanParameter():
                        neighbors = [not current_value]
                    case IntegerParameter(min_value=min_value, max_value=max_value):
                        if current_value != min_value:
                            neighbors.append(current_value - 1)
                        if current_value != max_value:
                            neighbors.append(current_value + 1)
                    case RealParameter(min_value=min_value, max_value=max_value):
                        epsilon = (max_value - min_value) / 1000  # TODO: Adjust epsilon dynamically.
                        if current_value > min_value:
                            neighbors.append(max(current_value - epsilon, min_value))
                        if current_value < max_value:
                            neighbors.append(min(current_value + epsilon, max_value))
                    case OrdinalParameter(values=values):
                        i = values.index(current_value)
                        if i > 0:
                            neighbors.append(values[i - 1])
                        if i < len(values) - 1:
                            neighbors.append(values[i + 1])
                    case CategoricalParameter(values=values):
                        neighbors = values.copy()
                        neighbors.remove(current_value)
                    case PermutationParameter(size=size):
                        for i in range(size):
                            for j in range(i + 1, size):
                                swapped_values = current_value.copy()
                                swapped_values[i], swapped_values[j] = swapped_values[j], swapped_values[i]
                                neighbors.append(swapped_values)
                solution = current_solution.copy()
                for neighbor in neighbors:
                    solution[name] = neighbor

                    score = self.tuning_problem.cost(solution)
                    yield solution, score
                    if score < current_score:
                        current_solution = solution.copy()
                        current_score = score
                        change = True
            if not change:
                # If no neighbor has any improvement, we are stuck in an (local) optimum, try again from random point.
                current_solution = self.random_config()
                current_score = self.tuning_problem.cost(current_solution)
                yield current_solution, current_score
