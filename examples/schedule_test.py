import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy
import numpy as np
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
    evaluator = func.time_evaluator(func.entry_name, dev, repeat=1)
    result = evaluator(a, b, c)
    # Check if calculation is correct
    correct_answer = numpy.dot(a.asnumpy(), b.asnumpy())
    c_numpyfied = c.asnumpy()
    assert np.allclose(
        c_numpyfied, correct_answer
    )  # test if same shape, elements have close enough values

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
    return result


def find_blocked_baseline() -> float:
    env = create_schedule()
    # Apply blocking as described in https://tvm.apache.org/docs/v0.13.0/how_to/optimize_operators/opt_gemm.html
    C = env.computed_arg
    s = env.schedule
    bn = 32
    kfactor = 4
    mo, no, mi, ni = s[C].tile(C.op.axis[0], C.op.axis[1], bn, bn)
    (kaxis,) = s[C].op.reduce_axis
    ko, ki = s[C].split(kaxis, factor=kfactor)

    # Hoist reduction domain outside the blocking loop
    s[C].reorder(mo, no, ko, ki, mi, ni)

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
    return result


if __name__ == "__main__":
    baseline_score = find_baseline()
    blocked_baseline_score = find_blocked_baseline()
    print("Baseline:", baseline_score)
    print("Baseline with blocking:", blocked_baseline_score)

    tuner = RandomSearch(check_constraints=False)
    param = ScheduleParam(create_schedule, finish_schedule, 1, 5)
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
    plt.figure()
    plt.plot(xs, means, label="Random Search")
    plt.fill_between(xs, mins, maxs, alpha=0.3)
    plt.plot(xs, [baseline_score] * len(xs), label="Baseline using Blocking")
    plt.plot(xs, [blocked_baseline_score] * len(xs), label="Baseline using Blocking")
    plt.xlabel("function evaluations")
    plt.ylabel("cost")
    plt.yscale("log")
    plt.legend()
    plt.show()
