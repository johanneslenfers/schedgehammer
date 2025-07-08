import numpy
import tvm
from tvm import te

from examples.schedules.tvm.tvm_schedule_problem import TVMScheduleProblem
from schedgehammer.param_types import ParamValue
from schedgehammer.schedules.schedule_type import ScheduleContext

N = 128
DTYPE = "float32"

class MttkrpProblem(TVMScheduleProblem):

    def __init__(self):
        super().__init__("mttkrp")

    def create_schedule(self) -> ScheduleContext:
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

    def cost_function(self, config: dict[str, ParamValue]) -> float:
        dev = tvm.device("llvm", 0)

        # Random generated tensor for testing
        a = tvm.nd.array(numpy.random.rand(N, N, N).astype(DTYPE), dev)
        b = tvm.nd.array(numpy.random.rand(N, N).astype(DTYPE), dev)
        c = tvm.nd.array(numpy.random.rand(N, N).astype(DTYPE), dev)
        func: tvm.module.Module = config["schedule"]
        evaluator = func.time_evaluator(func.entry_name, dev, repeat=3)
        result = evaluator(a, b, c).median
        return result

