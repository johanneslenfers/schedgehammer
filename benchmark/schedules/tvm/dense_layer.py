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

INPUT_LAYER_SIZE = 5096
OUTPUT_LAYER_SIZE = 2048
WEIGHTS_COUNT = INPUT_LAYER_SIZE * OUTPUT_LAYER_SIZE

DTYPE = "float32"


def create_dense_layer_schedule() -> ScheduleContext:
    layer1 = te.placeholder((INPUT_LAYER_SIZE,), name="layer1", dtype=DTYPE)
    weights = te.placeholder((WEIGHTS_COUNT,), name="weights", dtype=DTYPE)
    j = te.reduce_axis((0, INPUT_LAYER_SIZE), name="j")
    layer2 = te.compute(
        (OUTPUT_LAYER_SIZE,),
        lambda i: te.sum(layer1[j] * weights[i * INPUT_LAYER_SIZE + j], axis=j),
        name="layer2",
    )

    s = te.create_schedule(layer2.op)

    return ScheduleContext(
        [layer2.op.axis[0], j],
        {
            "schedule": s,
            "tensor": layer2,
            "alltensors": [layer1, weights, layer2],
        },
    )


def dense_layer_cost_function(config):
    dev = tvm.device("llvm", 0)
    layer1 = tvm.nd.array(numpy.random.rand(INPUT_LAYER_SIZE).astype(DTYPE), dev)
    weights = tvm.nd.array(numpy.random.rand(WEIGHTS_COUNT).astype(DTYPE), dev)
    layer2 = tvm.nd.array(numpy.zeros(OUTPUT_LAYER_SIZE, dtype=DTYPE), dev)

    func: tvm.module.Module = config["schedule"]
    evaluator = func.time_evaluator(func.entry_name, dev, repeat=1)

    result = evaluator(layer1, weights, layer2).mean

    # Check correctness of the result
    # layer1_np = layer1.asnumpy()
    # weights_np = weights.asnumpy().reshape(OUTPUT_LAYER_SIZE, INPUT_LAYER_SIZE)

    # # Compute expected result with numpy: dot product
    # dot_product = numpy.dot(weights_np, layer1_np)

    # layer2_np = layer2.asnumpy()
    # assert np.allclose(layer2_np, dot_product)
    return result


def get_ansor_dense_layer_results(iterations, runs):
    ansor_results = []

    @auto_scheduler.register_workload
    def create_task_func():
        layer1 = te.placeholder((INPUT_LAYER_SIZE,), name="layer1", dtype=DTYPE)
        weights = te.placeholder((WEIGHTS_COUNT,), name="weights", dtype=DTYPE)
        j = te.reduce_axis((0, INPUT_LAYER_SIZE), name="j")
        layer2 = te.compute(
            (OUTPUT_LAYER_SIZE,),
            lambda i: te.sum(layer1[j] * weights[i * INPUT_LAYER_SIZE + j], axis=j),
            name="layer2",
        )
        return [layer1, weights, layer2]

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
                auto_scheduler.RecordToFile("dense_layer.json"),
                StoreResultCallback(),
            ],
            verbose=2,
        )

        # Begin tuning process
        task.tune(tuning_options)
    return ansor_results
