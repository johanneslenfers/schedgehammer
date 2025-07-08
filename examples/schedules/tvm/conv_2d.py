import numpy
import tvm
from tvm import te

from examples.schedules.tvm.tvm_schedule_problem import TVMScheduleProblem
from schedgehammer.param_types import ParamValue
from schedgehammer.schedules.schedule_type import ScheduleContext

KERNEL_WIDTH = 9
INPUT_WIDTH = 2048

DTYPE = "float32"


def finish_schedule(ctx: ScheduleContext):
    return tvm.build(
        ctx.environment["schedule"],
        ctx.environment["alltensors"],
        "llvm",
    )


class Conv2DProblem(TVMScheduleProblem):

    def __init__(self):
        super().__init__("conv2d")

    def create_schedule(self):
        input = te.placeholder((INPUT_WIDTH, INPUT_WIDTH), name="input")
        kernel = te.placeholder((KERNEL_WIDTH, KERNEL_WIDTH), name="kernel")

        # Padding to match scipy's convolve2d with mode="same"
        # For kernel width K, padding is (K-1)//2 on each side
        pad_width = (KERNEL_WIDTH - 1) // 2

        # Define reduction axes for the convolution window
        rx = te.reduce_axis((0, KERNEL_WIDTH), name="rx")
        ry = te.reduce_axis((0, KERNEL_WIDTH), name="ry")

        output = te.compute(
            (INPUT_WIDTH, INPUT_WIDTH),
            lambda x, y: te.sum(
                te.if_then_else(
                    te.all(
                        x - pad_width + rx >= 0,
                        x - pad_width + rx < INPUT_WIDTH,
                        y - pad_width + ry >= 0,
                        y - pad_width + ry < INPUT_WIDTH,
                    ),
                    input[x - pad_width + rx, y - pad_width + ry] * kernel[rx, ry],
                    tvm.tir.const(0, DTYPE),
                ),
                axis=[rx, ry],
            ),
        )

        s = te.create_schedule(output.op)

        return ScheduleContext(
            [
                output.op.axis[0],
                output.op.axis[1],
                output.op.reduce_axis[0],
                output.op.reduce_axis[1],
            ],
            {
                "schedule": s,
                "tensor": output,
                "alltensors": [input, kernel, output],
            },
        )

    def cost_function(self, config: dict[str, ParamValue]) -> float:
        dev = tvm.device("llvm", 0)
        input = tvm.nd.array(numpy.random.rand(INPUT_WIDTH, INPUT_WIDTH).astype(DTYPE), dev)
        kernel = tvm.nd.array(
            numpy.random.rand(KERNEL_WIDTH, KERNEL_WIDTH).astype(DTYPE), dev
        )
        output = tvm.nd.array(numpy.zeros((INPUT_WIDTH, INPUT_WIDTH), dtype=DTYPE), dev)
        func: tvm.module.Module = config["schedule"]
        evaluator = func.time_evaluator(func.entry_name, dev, repeat=3)
        result = evaluator(input, kernel, output).median
        return result
