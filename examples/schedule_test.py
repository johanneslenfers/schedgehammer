# Only needed since this is in the same repo as schedgehammer.
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
#################################################################

from schedgehammer.param_types import ScheduleParam, MethodDescription, MethodParameter, SingleObjectParameter, \
    BasicParameter, ExpIntParam, ReturnObjectListParameter
from schedgehammer.problem import Problem
import tvm
import tvm.te
import numpy

def create_schedule():
    M = 1024
    K = 1024
    N = 1024
    k = te.reduce_axis((0, K), "k")
    A = te.placeholder((M, K), name="A")
    B = te.placeholder((K, N), name="B")
    C = te.compute((M, N), lambda m, n: te.sum(A[m, k] * B[k, n], axis=k), name="C")

    # Default schedule
    s = te.create_schedule(C.op)

    environment = {
        "schedule": s,
        "args": [A, B, C]
    }
    initial_values = {
        "axis": list(C.op.axis) + list(C.op.reduce_axis)
    }
    return environment, initial_values

def finish_schedule(environment):
    func = tvm.build(environment["schedule"], environment["args"], name="mmult")
    return func

def cost_function(config):
    M = 1024
    K = 1024
    N = 1024
    dtype = "float32"

    dev = tvm.device('llvm', 0)

    # Random generated tensor for testing
    a = tvm.nd.array(numpy.random.rand(M, K).astype(dtype), dev)
    b = tvm.nd.array(numpy.random.rand(K, N).astype(dtype), dev)
    c = tvm.nd.array(numpy.zeros((M, N), dtype=dtype), dev)

    func = config["schedule"]

    evaluator = func.time_evaluator(func.entry_name, dev, number=1)
    return evaluator(a, b, c).mean

def main():
    problem = Problem("schedule problem", {
        "schedule": ScheduleParam([
            MethodDescription(
                "tile",
                [
                    SingleObjectParameter("axis", remove_from_pool=True),
                    SingleObjectParameter("axis", remove_from_pool=True),
                    BasicParameter(ExpIntParam(2, 4, 10)),  # Or rather reference Top-Level tuned parameter?
                    BasicParameter(ExpIntParam(2, 4, 10)),
                ],
                ReturnObjectListParameter("axis"),
                lambda environment, args: list(environment['schedule'].tile(*args))
            ),
            # ...
        ],
            create_schedule,
            finish_schedule
        )
    }, cost_function)


if __name__ == '__main__':
    main()
