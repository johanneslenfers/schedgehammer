# Only needed since this is in the same repo as schedgehammer.
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
##############################################################
import copy

import numpy
import numpy as np
import tvm
from evaulate_schedule_language import evaluate_problem_for_schedule_language
from matplotlib import pyplot as plt
from tvm import auto_scheduler, te
from tvm.auto_scheduler.measure import PythonBasedMeasureCallback
from tvm_api import REORDER, SPLIT, TILE

from schedgehammer.problem import Problem
from schedgehammer.schedules.schedule_genetic_tuner import ScheduleGeneticTuner
from schedgehammer.schedules.schedule_random_search import ScheduleRandomSearch
from schedgehammer.schedules.schedule_type import ScheduleContext, ScheduleParam
from schedgehammer.tuner import EvalBudget, Tuner

N = 256
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
    evaluator = func.time_evaluator(func.entry_name, dev, repeat=1)
    result = evaluator(a, b, c).mean
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
                auto_scheduler.RecordToFile("mttkrp.json"),
                StoreResultCallback(),
            ],
            verbose=2,
        )

        # Begin tuning process
        task.tune(tuning_options)
    return ansor_results
