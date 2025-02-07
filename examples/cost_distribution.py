import json
import os
import sys
import threading

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import concurrent.futures

import numpy
import numpy as np
import stopit
import tvm
from matplotlib import pyplot as plt
from tvm import auto_scheduler, te
from tvm.auto_scheduler.measure import PythonBasedMeasureCallback

from examples.tvm_api import REORDER, SPLIT, TILE
from schedgehammer.genetic_tuner import GeneticTuner
from schedgehammer.problem import Problem
from schedgehammer.random_search import RandomSearch
from schedgehammer.schedule_type import ScheduleParam, ScheduleTree
from schedgehammer.tuner import EvalBudget

M = 512
K = 512
N = 512

DTYPE = "float32"

costs_per_schedule = []


def plot_results_from_several_runs(results, label) -> range:
    """Plot results from multiple runs and returns x values"""
    zipped_results = list(zip(*results))
    means = [np.mean(x) for x in zipped_results]
    mins = [np.min(x) for x in zipped_results]
    maxs = [np.max(x) for x in zipped_results]
    xs = range(len(zipped_results))
    plt.plot(xs, means, label=label)
    plt.fill_between(xs, mins, maxs, alpha=0.3)
    return xs


def create_schedule() -> ScheduleTree:
    k = te.reduce_axis((0, K), "k")
    A = te.placeholder((M, K), name="A")
    B = te.placeholder((K, N), name="B")
    C = te.compute((M, N), lambda m, n: te.sum(A[m, k] * B[k, n], axis=k), name="C")

    # Default schedule
    s = te.create_schedule(C.op)

    tree = ScheduleTree(
        schedule=s,
        computed_tensor=C,
        static_tensors=[A, B],
    )
    tree.add_original_axis(C.op.axis[0])
    tree.add_original_axis(C.op.axis[1])
    tree.add_original_axis(C.op.reduce_axis[0])
    return tree


def cost_function(config):
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

    costs_per_schedule[-1].append(result)
    return result


def finish_schedule(tree: ScheduleTree):
    return tvm.build(
        tree.schedule,
        tree.static_tensors + [tree.computed_tensor],
        name="anything",
    )


if __name__ == "__main__":
    param = ScheduleParam(
        create_schedule,
        finish_schedule,
        2,
        10,
        api_description=[TILE, SPLIT, REORDER],
        terminating_methods=[],
    )
    best_cost = float("inf")
    best_tree = None
    i = 0
    while not best_cost < 0.006 and i < 250:
        costs_per_schedule.append([])
        schedule = param.choose_random()
        tree = param.last_generated_tree
        cost = cost_function({"schedule": finish_schedule(tree)})
        if cost < best_cost:
            best_cost = cost
            best_tree = tree
        print(i, best_cost, cost)
        i += 1
    costs_per_schedule.append([])
    for variant_num in range(50):
        print(variant_num, "/", 40)
        best_tree.randomly_tweak_primitive_params()
        fresh_tree: ScheduleTree = create_schedule()
        best_tree.reapply_schedule(
            fresh_tree.schedule,
            fresh_tree.computed_tensor,
            fresh_tree.static_tensors,
            [axis.axis for axis in fresh_tree.original_axes],
        )
        cost = cost_function({"schedule": finish_schedule(best_tree)})
    plt.figure()
    # Plot  results as dot diagram
    for i, costs in enumerate(costs_per_schedule):
        plt.plot([i] * len(costs), costs, "o")
    plt.yscale("log")
    plt.show()
    print(str(best_tree))
