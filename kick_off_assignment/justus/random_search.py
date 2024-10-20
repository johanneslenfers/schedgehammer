import math

from tuner import Tuner


class RandomSearch(Tuner):

    def generate_tuning(self):
        best_solution = None
        best_score = math.inf

        while True:
            solution = self.random_config()
            score = self.tuning_problem.cost(solution)
            if score < best_score:
                best_solution = solution
                best_score = score
            yield best_solution, best_score
