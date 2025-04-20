import numpy
import tvm
from scipy.signal import convolve2d
from tvm import auto_scheduler, te
from tvm.auto_scheduler.measure import PythonBasedMeasureCallback

from schedgehammer.schedules.schedule_type import ScheduleContext

KERNEL_WIDTH = 5
INPUT_WIDTH = 1024

DTYPE = "float32"


def get_ansor_conv_2d_results(iterations, runs):
    ansor_results = []

    @auto_scheduler.register_workload
    def create_task_func():
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
        return [input, kernel, output]

    class StoreResultCallback(PythonBasedMeasureCallback):
        def callback(self, policy, inputs, results):
            for result in results[0:]:
                cost = float(result.costs[0])
                if not ansor_results[-1] or cost < ansor_results[-1][-1]:
                    ansor_results[-1].append(cost)
                else:
                    ansor_results[-1].append(ansor_results[-1][-1])

    # Create the search task
    target = tvm.target.Target("llvm")
    for _ in range(runs):
        ansor_results.append([])
        task = auto_scheduler.SearchTask(func=create_task_func, target=target)

        tuning_options = auto_scheduler.TuningOptions(
            num_measure_trials=min(iterations, 63),
            measure_callbacks=[
                auto_scheduler.RecordToFile("conv_2d.json"),
                StoreResultCallback(),
            ],
            verbose=2,
        )

        # Begin tuning process
        task.tune(tuning_options)
    return ansor_results


def create_2d_conv_schedule() -> ScheduleContext:
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


def conv_2d_cost_function(config):
    dev = tvm.device("llvm", 0)
    input = tvm.nd.array(numpy.random.rand(INPUT_WIDTH, INPUT_WIDTH).astype(DTYPE), dev)
    kernel = tvm.nd.array(
        numpy.random.rand(KERNEL_WIDTH, KERNEL_WIDTH).astype(DTYPE), dev
    )
    output = tvm.nd.array(numpy.zeros((INPUT_WIDTH, INPUT_WIDTH), dtype=DTYPE), dev)
    func: tvm.module.Module = config["schedule"]
    evaluator = func.time_evaluator(func.entry_name, dev, repeat=1)
    result = evaluator(input, kernel, output).mean

    # input_np = input.asnumpy()
    # kernel_np = kernel.asnumpy()
    # output_np = output.asnumpy()

    # reference_output = numpy.zeros_like(input_np)
    # pad_width = (KERNEL_WIDTH - 1) // 2

    # for x in range(INPUT_WIDTH):
    #     for y in range(INPUT_WIDTH):
    #         acc = 0.0
    #         for rx in range(KERNEL_WIDTH):
    #             for ry in range(KERNEL_WIDTH):
    #                 in_x = x - pad_width + rx
    #                 in_y = y - pad_width + ry
    #                 if 0 <= in_x < INPUT_WIDTH and 0 <= in_y < INPUT_WIDTH:
    #                     acc += input_np[in_x, in_y] * kernel_np[rx, ry]
    #         reference_output[x, y] = acc
    # assert numpy.allclose(output_np, reference_output, rtol=1e-5, atol=1e-5)
    print("Cost:", result)
    return result
