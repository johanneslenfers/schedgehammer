import json
import sys

import numpy
import numpy as np
import tvm
from matplotlib import pyplot as plt
from tvm import auto_scheduler, te
from tvm.auto_scheduler.measure import PythonBasedMeasureCallback

from schedgehammer.benchmark import benchmark
from schedgehammer.param_types import ParamValue
from tvm_api import REORDER, SPLIT, TILE

from schedgehammer.problem import Problem
from schedgehammer.genetic_tuner_2 import GeneticTuner2
from schedgehammer.random_search_2 import RandomSearch2
from schedgehammer.schedules.schedule_type import ScheduleContext, ScheduleParam
from schedgehammer.tuner import EvalBudget

M = 1024
K = 1024
N = 1024

DTYPE = "float32"
ITERATIONS = 200  # If >63 limit ansors iterations else it will crash

genetic_results = []
random_results = []


def plot_results_from_several_runs(results, label) -> range:
    """Plot results from multiple runs and returns x values"""
    zipped_results = list(zip(*results))
    means = [np.mean(x) for x in zipped_results]
    lower_percentile = [np.percentile(x, 5) for x in zipped_results]
    upper_percentile = [np.percentile(x, 95) for x in zipped_results]
    xs = range(len(zipped_results))
    plt.plot(xs, means, label=label)
    plt.fill_between(
        xs,
        lower_percentile,
        upper_percentile,
        alpha=0.3,
    )
    return xs

def create_schedule() -> ScheduleContext:
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


def cost_function(config):
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

    print("COST:", result)
    return result


def finish_schedule(ctx: ScheduleContext):
    return tvm.build(
        ctx.environment["schedule"],
        ctx.environment["alltensors"],
        "llvm --opt-level=0",
        name="anything",
    )


def get_ansor_results(iterations, runs):
    ansor_results = []

    class StoreResultCallback(PythonBasedMeasureCallback):
        def callback(self, policy, inputs, results):
            for result in results[0:]:
                ansor_results[-1].append(result.costs[0])

    @auto_scheduler.register_workload
    def create_task_func():
        k = te.reduce_axis((0, K), "k")
        A = te.placeholder((M, K), name="A")
        B = te.placeholder((K, N), name="B")
        C = te.compute((M, N), lambda m, n: te.sum(A[m, k] * B[k, n], axis=k), name="C")
        return [A, B, C]

    # Create the search task
    target = tvm.target.Target("llvm --opt-level=0")
    for _ in range(runs):
        ansor_results.append([])
        task = auto_scheduler.SearchTask(func=create_task_func, target=target)

        tuning_options = auto_scheduler.TuningOptions(
            num_measure_trials=iterations,
            measure_callbacks=[
                auto_scheduler.RecordToFile("matmul.json"),
                StoreResultCallback(),
            ],
            verbose=2,
        )

        # Begin tuning process
        task.tune(tuning_options)
    return ansor_results

def get_baseline() -> float:
    # Find time of unchanged schedule
    env = create_schedule().environment
    func = tvm.build(env["schedule"], env["alltensors"], "llvm --opt-level=0", name="anything")
    dev = tvm.device("llvm --opt-level=0", 0)
    a = tvm.nd.array(numpy.random.rand(M, K).astype(DTYPE), dev)
    b = tvm.nd.array(numpy.random.rand(K, N).astype(DTYPE), dev)
    c = tvm.nd.array(numpy.zeros((M, N), dtype=DTYPE), dev)
    evaluator = func.time_evaluator(func.entry_name, dev, repeat=3, number=3)
    result = evaluator(a, b, c).median
    return result


def get_blocking_baseline(bn=32, kfactor=4) -> float:
    env = create_schedule().environment
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

class MMProblem(Problem):

    def __init__(self):
        param = ScheduleParam(
            create_schedule,
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
        return cost_function(config)



if __name__ == "__main__":
    if sys.argv[1] == 'ansor':
        with open('results/ansor/mm.json', 'w') as f:
            json.dump(get_ansor_results(63, 25), f)
    else:
        benchmark(
            MMProblem,
            [EvalBudget(ITERATIONS)],
            {
                "genetic_tuner": GeneticTuner2(),
                "random_tuner": RandomSearch2(),
            },
            f"results/tvm/mm",
            15,
            True,
            16,
        )

        baseline_score = get_baseline()
        print("Baseline:", baseline_score)

        best_block_schedule = float("inf")
        for bn_exp in range(1, 10):
            for kfactor_exp in range(1, 10):
                v = get_blocking_baseline(2**bn_exp, 2**kfactor_exp)
                print(
                    "Try default schedule with hyperparameters:", 2**bn_exp, 2**kfactor_exp, v
                )
                best_block_schedule = min(best_block_schedule, v)
        print("Best block schedule:", best_block_schedule)
