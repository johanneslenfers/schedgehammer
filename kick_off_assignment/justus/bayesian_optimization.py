import math
from typing import Generator, Tuple, Dict, List
import numpy as np
import scipy

from local_search import LocalSearch
from tuning_problem import TuningConfig, TuningProblem
from tuner import Tuner


class BayesianOptimization(Tuner):

    initial_sample_size: int

    def __init__(self, tuning_problem: TuningProblem, initial_sample_size: int = 10):
        super().__init__(tuning_problem)
        self.initial_sample_size = initial_sample_size

    def distance(self, config1: TuningConfig, config2: TuningConfig, thetas: Dict[str, float], ps: Dict[str, float]):
        s = 0
        for name in self.tuning_problem.parameters:
            d = self.tuning_problem.parameters[name].distance(config1[name], config2[name])
            s += thetas[name] * (d ** ps[name])
        return s

    def correlation_matrix(self, values: List[TuningConfig], thetas: Dict[str, float], ps: Dict[str, float]):
        n = len(values)
        matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i < j:
                    matrix[i, j] = math.exp(-self.distance(values[i], values[j], thetas, ps))

        return matrix + matrix.T + np.identity(n) * 1.001  # Make symmetric matrix.

    def likelihood_func(self, parameters, xs, ys):
        n = len(self.tuning_problem.parameters)
        ones = np.ones(ys.shape)
        thetas = dict(zip(self.tuning_problem.parameters.keys(), parameters[:n]))
        ps = dict(zip(self.tuning_problem.parameters.keys(), parameters[n:]))
        r = self.correlation_matrix(xs, thetas, ps)
        r_inv = scipy.linalg.inv(r)
        mean = (ones.T @ r_inv @ ys) / (ones.T @ r_inv @ ones)
        part_thing = (ys - mean * ones).T @ r_inv @ (ys - mean * ones)
        stddev = part_thing / self.initial_sample_size
        return (math.exp(part_thing / (2 * stddev)) /
                ((2 * math.pi) ** (n / 2) * stddev ** (n / 2)))  # theoretically needs sqrt(det(R)), but that messes stuff up.

    def evaluate_in_model(self, config, xs, ys, mean, stddev, r_inv, thetas, ps, f_min):
        dist = []
        ones = np.ones(ys.shape)
        for x in xs:
            dist.append(math.exp(-self.distance(config, x, thetas, ps)))
        dist = np.array(dist)
        this_mean = mean + dist.T @ r_inv @ (ys - mean)
        this_stddev = stddev * (1 - dist.T @ r_inv @ dist + ((1 - ones.T @ r_inv @ dist) ** 2) / (ones.T @ r_inv @ ones))
        this_variance = math.sqrt(abs(this_stddev))
        ei = (f_min - this_mean) * scipy.stats.norm.pdf((f_min - this_mean) / this_variance) + this_variance * scipy.stats.norm.cdf((f_min - this_mean) / this_variance)
        return ei


    def generate_tuning(self) -> Generator[Tuple[TuningConfig, float], None, None]:
        n = len(self.tuning_problem.parameters)
        xs = []
        ys = []
        for _ in range(self.initial_sample_size):
            x = self.random_config()
            xs.append(x)
            y = self.tuning_problem.cost(x)
            ys.append(y)
            yield x, y

        best_score = min(ys)
        best_solution = xs[ys.index(best_score)]

        ys = np.array(ys)
        ones = np.ones(ys.shape)

        solution = scipy.optimize.minimize(lambda a, b, c: -self.likelihood_func(a, b, c),
                                x0=np.array([0.1] * n + [2] * n),
                                method='Nelder-Mead',
                                bounds=[(0, math.inf)] * n + [(1, 2)] * n,
                                args=(xs, ys))
        thetas = dict(zip(self.tuning_problem.parameters.keys(), solution['x'][:n]))
        ps = dict(zip(self.tuning_problem.parameters.keys(), solution['x'][n:]))

        r = self.correlation_matrix(xs, thetas, ps)
        r_inv = np.linalg.inv(r)
        mean = (ones.T @ r_inv @ ys) / (ones.T @ r_inv @ ones)
        stddev = ((ys - mean * ones).T @ r_inv @ (ys - mean * ones)) / self.initial_sample_size

        while True:
            search = LocalSearch(TuningProblem(self.tuning_problem.parameters,
                                      lambda value: -self.evaluate_in_model(value, xs, ys, mean, stddev, r_inv, thetas, ps, f_min=best_score)))
            solution, expected_score = search.num_evaluations(1000, False)
            score = self.tuning_problem.cost(solution)
            yield solution, score
            print(solution, expected_score, score)
            if score < best_score:
                best_score = score
                best_solution = solution
            xs.append(solution)
            ys = np.append(ys, score)
            r = self.correlation_matrix(xs, thetas, ps)
            r_inv = np.linalg.inv(r)
