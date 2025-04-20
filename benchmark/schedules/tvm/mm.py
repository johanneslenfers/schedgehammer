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

M = 512
K = 512
N = 512

DTYPE = "float32"


def create_mm_schedule() -> ScheduleContext:
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


def mm_cost_function(config):
    dev = tvm.device("llvm", 0)

    # Random generated tensor for testing
    a = tvm.nd.array(numpy.random.rand(M, K).astype(DTYPE), dev)
    b = tvm.nd.array(numpy.random.rand(K, N).astype(DTYPE), dev)
    c = tvm.nd.array(numpy.zeros((M, N), dtype=DTYPE), dev)

    func: tvm.module.Module = config["schedule"]
    evaluator = func.time_evaluator(func.entry_name, dev, repeat=1)

    result = evaluator(a, b, c).mean
    correct_answer = numpy.dot(a.asnumpy(), b.asnumpy())
    c_numpyfied = c.asnumpy()
    assert np.allclose(
        c_numpyfied, correct_answer
    )  # test if same shape, elements have close enough values

    return result


def get_ansor_mm_results(iterations, runs):
    ansor_results = []

    @auto_scheduler.register_workload
    def create_task_func():
        k = te.reduce_axis((0, K), "k")
        A = te.placeholder((M, K), name="A")
        B = te.placeholder((K, N), name="B")
        C = te.compute((M, N), lambda m, n: te.sum(A[m, k] * B[k, n], axis=k), name="C")
        return [A, B, C]

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
                auto_scheduler.RecordToFile("matmul.json"),
                StoreResultCallback(),
            ],
            verbose=2,
        )

        # Begin tuning process
        task.tune(tuning_options)
    return ansor_results


def get_blocking_baseline(bn=32, kfactor=4) -> float:
    env = create_mm_schedule().environment
    # Apply blocking as described in https://tvm.apache.org/docs/v0.13.0/how_to/optimize_operators/opt_gemm.html
    C = env["tensor"]
    s = env["schedule"]
    mo, no, mi, ni = s[C].tile(C.op.axis[0], C.op.axis[1], bn, bn)
    (kaxis,) = s[C].op.reduce_axis
    ko, ki = s[C].split(kaxis, factor=kfactor)

    s[C].reorder(mo, no, ko, ki, mi, ni)

    func = tvm.build(s, env["alltensors"], name="anything")
    dev = tvm.device("llvm", 0)
    a = tvm.nd.array(numpy.random.rand(M, K).astype(DTYPE), dev)
    b = tvm.nd.array(numpy.random.rand(K, N).astype(DTYPE), dev)
    c = tvm.nd.array(numpy.zeros((M, N), dtype=DTYPE), dev)
    evaluator = func.time_evaluator(func.entry_name, dev, repeat=1)
    result = evaluator(a, b, c).mean
    return result


def get_best_mm_blocking_baseline() -> float:
    best_blocking_baseline = float("inf")
    for bn_exp in range(1, 10):
        for kfactor_exp in range(1, 10):
            best_blocking_baseline = min(
                best_blocking_baseline, get_blocking_baseline(2**bn_exp, 2**kfactor_exp)
            )
    return best_blocking_baseline
