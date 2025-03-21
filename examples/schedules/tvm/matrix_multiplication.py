import numpy
import tvm
from tvm import te

from tvm_api import REORDER, SPLIT, TILE
from schedgehammer.problem import Problem
from schedgehammer.schedules.schedule_type import ScheduleParam, ScheduleContext

M = 1024
K = 1024
N = 1024

DTYPE = "float32"
RUNS = 3

def create_schedule() -> ScheduleContext:
    k = te.reduce_axis((0, K), "k")
    A = te.placeholder((M, K), name="A")
    B = te.placeholder((K, N), name="B")
    C = te.compute((M, N), lambda m, n: te.sum(A[m, k] * B[k, n], axis=k), name="C")

    # Default schedule
    s = te.create_schedule(C.op)

    return ScheduleContext(
        [C.op.axis[0], C.op.axis[1], C.op.reduce_axis[0]],
        {
            'schedule': s,
            'tensor': C,
            'alltensors': [A, B, C],
        }
    )


def cost_function(config):
    dev = tvm.device("llvm", 0)

    # Random generated tensor for testing
    a = tvm.nd.array(numpy.random.rand(M, K).astype(DTYPE), dev)
    b = tvm.nd.array(numpy.random.rand(K, N).astype(DTYPE), dev)
    c = tvm.nd.array(numpy.zeros((M, N), dtype=DTYPE), dev)

    func: tvm.module.Module = config["schedule"]
    evaluator = func.time_evaluator(func.entry_name, dev, repeat=1, number=5)
    result = evaluator(a, b, c).mean

    return result


def finish_schedule(ctx: ScheduleContext):
    return tvm.build(
        ctx.environment['schedule'],
        ctx.environment['alltensors'],
        name="anything",
    )

mm_schedule = ScheduleParam(
    create_schedule,
    finish_schedule,
    0,
    4,
    api_description=[TILE, SPLIT, REORDER],
)

mm_problem = Problem(
    "mm",
    {"schedule": mm_schedule},
    cost_function,
    [],
    init_solver=False
)
