# Adapter to provide mttkrp functions for performance_distribution.py
import numpy
import tvm
from tvm import te

from schedgehammer.schedules.schedule_type import ScheduleContext

N = 128
DTYPE = "float32"


def create_mttkrp_schedule() -> ScheduleContext:
    """Create a TVM schedule for MTTKRP operation."""
    k = te.reduce_axis((0, N), name="k")
    l = te.reduce_axis((0, N), name="l")
    A = te.placeholder((N, N, N), name="A")
    B = te.placeholder((N, N), name="B")
    C = te.placeholder((N, N), name="C")
    out = te.compute(
        (N, N),
        lambda i, j: te.sum(A[i, k, l] * B[l, j] * C[k, j], axis=[k, l]),
        name="out",
    )

    s = te.create_schedule(out.op)

    return ScheduleContext(
        [out.op.axis[0], out.op.axis[1], out.op.reduce_axis[0], out.op.reduce_axis[1]],
        {"schedule": s, "tensor": out, "alltensors": [A, B, C]},
    )


def mttkrp_cost_function(config) -> float:
    """Cost function for MTTKRP schedule evaluation."""
    dev = tvm.device("llvm", 0)

    # Random generated tensor for testing
    a = tvm.nd.array(numpy.random.rand(N, N, N).astype(DTYPE), dev)
    b = tvm.nd.array(numpy.random.rand(N, N).astype(DTYPE), dev)
    c = tvm.nd.array(numpy.random.rand(N, N).astype(DTYPE), dev)
    func: tvm.module.Module = config["schedule"]
    evaluator = func.time_evaluator(func.entry_name, dev, repeat=3)
    result = evaluator(a, b, c).median
    return result
