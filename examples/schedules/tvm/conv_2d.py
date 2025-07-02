import json
import sys

import numpy
import tvm
from tvm import auto_scheduler, te
from tvm.auto_scheduler import HardwareParams
from tvm.auto_scheduler.measure import PythonBasedMeasureCallback

from tvm_api import TILE, SPLIT, REORDER
from schedgehammer.benchmark import benchmark
from schedgehammer.param_types import ParamValue
from schedgehammer.problem import Problem
from schedgehammer.schedules.schedule_genetic_tuner import ScheduleGeneticTuner
from schedgehammer.schedules.schedule_random_search import ScheduleRandomSearch
from schedgehammer.schedules.schedule_type import ScheduleContext, ScheduleParam
from schedgehammer.tuner import EvalBudget

KERNEL_WIDTH = 9
INPUT_WIDTH = 2048

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
                ansor_results[-1].append(float(result.costs[0]))

    # Create the search task
    target = tvm.target.Target("llvm")
    for _ in range(runs):
        ansor_results.append([])
        task = auto_scheduler.SearchTask(
            func=create_task_func,
            target=target,
            hardware_params=HardwareParams(num_cores=1, target=target)
        )

        tuning_options = auto_scheduler.TuningOptions(
            num_measure_trials=iterations,
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
    evaluator = func.time_evaluator(func.entry_name, dev, repeat=3)
    result = evaluator(input, kernel, output).median
    return result

def finish_schedule(ctx: ScheduleContext):
    return tvm.build(
        ctx.environment["schedule"],
        ctx.environment["alltensors"],
        "llvm",
    )


class Conv2DProblem(Problem):

    def __init__(self):
        param = ScheduleParam(
            create_2d_conv_schedule,
            finish_schedule,
            2,
            10,
            api_description=[TILE, SPLIT, REORDER],
        )

        super().__init__(
            "schedge",
            {"schedule": param},
            [],
            init_solver=False,
        )

    def cost_function(self, config: dict[str, ParamValue]) -> float:
        return conv_2d_cost_function(config)

if __name__ == "__main__":
    if sys.argv[1] == 'ansor':
        with open('results/ansor/conv2d.json', 'w') as f:
            json.dump(get_ansor_conv_2d_results(63, 25), f)
    else:
        benchmark(
            Conv2DProblem,
            [EvalBudget(100)],
            {
                "genetic_tuner": ScheduleGeneticTuner(),
                "random_tuner": ScheduleRandomSearch(),
            },
            f"results/tvm/conv2d",
            15,
            True,
            16,
        )
