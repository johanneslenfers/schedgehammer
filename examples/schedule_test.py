import os
import sys
import time

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy
import tvm
from matplotlib import pyplot as plt
from tvm import te

from schedgehammer.problem import Problem
from schedgehammer.random_search import RandomSearch
from schedgehammer.schedule_type import ScheduleEnvironment, ScheduleParam
from schedgehammer.tuner import EvalBudget

M = 512
K = 512
N = 512

ITERATIONS = 30

results = []


def create_schedule() -> ScheduleEnvironment:
    k = te.reduce_axis((0, K), "k")
    A = te.placeholder((M, K), name="A")
    B = te.placeholder((K, N), name="B")
    C = te.compute((M, N), lambda m, n: te.sum(A[m, k] * B[k, n], axis=k), name="C")

    # Default schedule
    s = te.create_schedule(C.op)

    return ScheduleEnvironment(
        schedule=s,
        static_args=[A, B],
        computed_arg=C,
        axis_pool=list(C.op.axis) + list(C.op.reduce_axis),
    )


def cost_function(config):
    dtype = "float32"

    dev = tvm.device("llvm", 0)

    # Random generated tensor for testing
    a = tvm.nd.array(numpy.random.rand(M, K).astype(dtype), dev)
    b = tvm.nd.array(numpy.random.rand(K, N).astype(dtype), dev)
    c = tvm.nd.array(numpy.zeros((M, N), dtype=dtype), dev)

    func = config["schedule"]

    evaluator = func.time_evaluator(func.entry_name, dev, repeat=3)
    result = evaluator(a, b, c)
    # Check if calculation is correct
    assert c.asnumpy().all() == numpy.dot(a.asnumpy(), b.asnumpy()).all()
    print("Result:", result)
    if not results or result.mean < results[-1]["mean"]:
        results.append({"min": result.min, "mean": result.mean, "max": result.max})
    else:
        results.append(results[-1])
    return result.mean


def finish_schedule(env: ScheduleEnvironment):
    return tvm.build(
        env.schedule, env.static_args + [env.computed_arg], name="anything"
    )


def find_baseline() -> float:
    # Find time of unchanged schedule
    env = create_schedule()
    func = tvm.build(
        env.schedule, env.static_args + [env.computed_arg], name="anything"
    )
    dtype = "float32"
    dev = tvm.device("llvm", 0)
    a = tvm.nd.array(numpy.random.rand(M, K).astype(dtype), dev)
    b = tvm.nd.array(numpy.random.rand(K, N).astype(dtype), dev)
    c = tvm.nd.array(numpy.zeros((M, N), dtype=dtype), dev)
    evaluator = func.time_evaluator(func.entry_name, dev, repeat=3)
    result = evaluator(a, b, c).mean
    print("Baseline:", result)
    return result


if __name__ == "__main__":
    baseline_score = find_baseline()
    tuner = RandomSearch(check_constraints=False)
    param = ScheduleParam(create_schedule, finish_schedule, 1, 4)
    tuner.tune(
        problem=Problem(
            "schedge",
            {"schedule": param},
            cost_function,
            [],
        ),
        budgets=[EvalBudget(ITERATIONS)],
    )
    means, mins, maxs = zip(*[(x["mean"], x["min"], x["max"]) for x in results])
    xs = range(len(means))
    print("Failed schedule generations:", param.failed_random_generations)
    print("Successful schedule generations:", param.successful_random_generations)
    plt.figure()
    plt.plot(xs, means, label="Random Search")
    plt.fill_between(xs, mins, maxs, alpha=0.3)
    plt.plot(xs, [baseline_score] * len(xs), label="baseline")
    plt.xlabel("function evaluations")
    plt.ylabel("cost")
    plt.yscale("log")
    plt.legend()
    plt.show()
