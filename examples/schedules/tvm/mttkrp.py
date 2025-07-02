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

N = 128
DTYPE = "float32"

def create_mttkrp_schedule() -> ScheduleContext:
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


def mttkrp_cost_function(config):
    dev = tvm.device("llvm", 0)

    # Random generated tensor for testing
    a = tvm.nd.array(numpy.random.rand(N, N, N).astype(DTYPE), dev)
    b = tvm.nd.array(numpy.random.rand(N, N).astype(DTYPE), dev)
    c = tvm.nd.array(numpy.random.rand(N, N).astype(DTYPE), dev)
    func: tvm.module.Module = config["schedule"]
    evaluator = func.time_evaluator(func.entry_name, dev, repeat=3)
    result = evaluator(a, b, c).median
    return result


def get_ansor_mttkrp_results(iterations, runs):
    ansor_results = []

    @auto_scheduler.register_workload
    def create_task_func():
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
        return [A, B, C, out]

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
                auto_scheduler.RecordToFile("mttkrp.json"),
                StoreResultCallback(),
            ],
            verbose=2,
        )

        # Begin tuning process
        task.tune(tuning_options)
    return ansor_results

def finish_schedule(ctx: ScheduleContext):
    return tvm.build(
        ctx.environment["schedule"],
        ctx.environment["alltensors"],
        "llvm",
    )


class MttkrpProblem(Problem):

    def __init__(self):
        param = ScheduleParam(
            create_mttkrp_schedule,
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
        return mttkrp_cost_function(config)

if __name__ == "__main__":
    if sys.argv[1] == 'ansor':
        with open('results/ansor/mttkrp.json', 'w') as f:
            json.dump(get_ansor_mttkrp_results(63, 25), f)
    else:
        benchmark(
            MttkrpProblem,
            [EvalBudget(100)],
            {
                "genetic_tuner": ScheduleGeneticTuner(),
                "random_tuner": ScheduleRandomSearch(),
            },
            f"results/tvm/mttkrp",
            15,
            True,
            16,
        )
