import os
import sys
import time

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy
import tvm
from tvm import te

from schedgehammer.problem import Problem
from schedgehammer.random_search import RandomSearch
from schedgehammer.schedule_type import ScheduleEnvironment, ScheduleParam
from schedgehammer.tuner import TuningAttempt

M = 1024
K = 1024
N = 1024


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

    evaluator = func.time_evaluator(func.entry_name, dev, number=5)
    result = evaluator(a, b, c).mean
    # Check if calculation is correct
    assert c.asnumpy().all() == numpy.dot(a.asnumpy(), b.asnumpy()).all()
    print("Result:", result)
    return result


def find_baseline():
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
    evaluator = func.time_evaluator(func.entry_name, dev, number=5)
    result = evaluator(a, b, c).mean
    print("Baseline:", result)
    return result


if __name__ == "__main__":
    find_baseline()
    tuner = RandomSearch(check_constraints=False)
    tuner.do_tuning(
        TuningAttempt(
            problem=Problem(
                "schedge",
                {"schedule": ScheduleParam(create_schedule, 1, 4)},
                cost_function,
                [],
            ),
            budgets=[],
        )
    )
