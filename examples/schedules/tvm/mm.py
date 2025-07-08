import numpy
import numpy as np
import tvm
from tvm import te

from examples.schedules.tvm.tvm_schedule_problem import TVMScheduleProblem
from schedgehammer.param_types import ParamValue
from schedgehammer.schedules.schedule_type import ScheduleContext

M = 1024
K = 1024
N = 1024

DTYPE = "float32"

class MMProblem(TVMScheduleProblem):

    def __init__(self):
        a = numpy.random.rand(M, K).astype(DTYPE)
        b = numpy.random.rand(K, N).astype(DTYPE)
        c = numpy.zeros((M, N), dtype=DTYPE)
        super().__init__("mm")

    def create_schedule(self) -> ScheduleContext:
        k = te.reduce_axis((0, K), "k")
        A = te.placeholder((M, K), name="A")
        B = te.placeholder((K, N), name="B")
        C = te.compute((M, N), lambda m, n: te.sum(A[m, k] * B[k, n], axis=k), name="C")

        # Default schedule
        s = te.create_schedule(C.op)

        return ScheduleContext(
            [C.op.axis[0], C.op.axis[1], C.op.reduce_axis[0]],
            {
                "schedule": s,
                "tensor": C,
                "alltensors": [A, B, C],
            },
        )

    def cost_function(self, config: dict[str, ParamValue]) -> float:
        dev = tvm.device("llvm --opt-level=0", 0)

        # Random generated tensor for testing
        a = tvm.nd.array(numpy.random.rand(M, K).astype(DTYPE), dev)
        b = tvm.nd.array(numpy.random.rand(K, N).astype(DTYPE), dev)
        c = tvm.nd.array(numpy.zeros((M, N), dtype=DTYPE), dev)

        func: tvm.module.Module = config["schedule"]
        evaluator = func.time_evaluator(func.entry_name, dev, repeat=3, number=3)
        result = evaluator(a, b, c).median

        correct_answer = numpy.dot(a.asnumpy(), b.asnumpy())
        c_numpyfied = c.asnumpy()
        assert np.allclose(
            c_numpyfied, correct_answer
        )  # test if same shape, elements have close enough values

        return result

    def get_blocking_baseline(self, bn=32, kfactor=4) -> float:
        env = self.create_schedule().environment
        # Apply blocking as described in https://tvm.apache.org/docs/v0.13.0/how_to/optimize_operators/opt_gemm.html
        C = env["tensor"]
        s = env["schedule"]
        mo, no, mi, ni = s[C].tile(C.op.axis[0], C.op.axis[1], bn, bn)
        (kaxis,) = s[C].op.reduce_axis
        ko, ki = s[C].split(kaxis, factor=kfactor)

        s[C].reorder(mo, no, ko, ki, mi, ni)

        func = tvm.build(s, env["alltensors"], "llvm --opt-level=0", name="anything")
        dev = tvm.device("llvm --opt-level=0", 0)
        a = tvm.nd.array(numpy.random.rand(M, K).astype(DTYPE), dev)
        b = tvm.nd.array(numpy.random.rand(K, N).astype(DTYPE), dev)
        c = tvm.nd.array(numpy.zeros((M, N), dtype=DTYPE), dev)
        evaluator = func.time_evaluator(func.entry_name, dev, repeat=3, number=3)
        result = evaluator(a, b, c).median
        return result

    def get_best_blocking_baseline(self) -> float:
        best_block_schedule = float("inf")
        for bn_exp in range(1, 10):
            for kfactor_exp in range(1, 10):
                v = self.get_blocking_baseline(2 ** bn_exp, 2 ** kfactor_exp)
                print("Try default schedule with hyperparameters:", 2 ** bn_exp, 2 ** kfactor_exp, v)
                best_block_schedule = min(best_block_schedule, v)
        print("Best block schedule:", best_block_schedule)
        return best_block_schedule
