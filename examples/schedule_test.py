import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy
import numpy as np
import tvm
from matplotlib import pyplot as plt
from tvm import auto_scheduler, te

from schedgehammer.problem import Problem
from schedgehammer.random_search import RandomSearch
from schedgehammer.schedule_type import ScheduleEnvironment, ScheduleParam
from schedgehammer.tuner import EvalBudget

M = 512
K = 512
N = 512

DTYPE = "float32"
ITERATIONS = 50
RUNS = 3

results = []


def create_schedule() -> ScheduleEnvironment:
    k = te.reduce_axis((0, K), "k")
    A = te.placeholder((M, K), name="A")
    B = te.placeholder((K, N), name="B")
    C = te.compute((M, N), lambda m, n: te.sum(A[m, k] * B[k, n], axis=k), name="C")

    # Default schedule
    s = te.create_schedule(C.op)

    return ScheduleEnvironment(
        schedule=s,
        static_args=[A, B],
        computed_arg=C,
        axis_pool=list(C.op.axis) + list(C.op.reduce_axis),
    )


def cost_function(config):
    dev = tvm.device("llvm", 0)

    # Random generated tensor for testing
    a = tvm.nd.array(numpy.random.rand(M, K).astype(DTYPE), dev)
    b = tvm.nd.array(numpy.random.rand(K, N).astype(DTYPE), dev)
    c = tvm.nd.array(numpy.zeros((M, N), dtype=DTYPE), dev)

    func = config["schedule"]
    evaluator = func.time_evaluator(func.entry_name, dev, repeat=1)
    result = evaluator(a, b, c)
    # Check if calculation is correct
    correct_answer = numpy.dot(a.asnumpy(), b.asnumpy())
    c_numpyfied = c.asnumpy()
    assert np.allclose(
        c_numpyfied, correct_answer
    )  # test if same shape, elements have close enough values

    print("Result:", result)
    if not results[-1] or result.mean < results[-1][-1]:
        results[-1].append(result.mean)
    else:
        results[-1].append(results[-1][-1])
    return result.mean


def finish_schedule(env: ScheduleEnvironment):
    return tvm.build(
        env.schedule, env.static_args + [env.computed_arg], name="anything"
    )


def get_ansor_baseline() -> float:
    @auto_scheduler.register_workload
    def create_task_func():
        k = te.reduce_axis((0, K), "k")
        A = te.placeholder((M, K), name="A")
        B = te.placeholder((K, N), name="B")
        C = te.compute((M, N), lambda m, n: te.sum(A[m, k] * B[k, n], axis=k), name="C")
        return [A, B, C]

    # Create the search task
    target = tvm.target.Target("llvm")
    task = auto_scheduler.SearchTask(func=create_task_func, target=target)

    # Set search policy and performance measurement options
    tuning_options = auto_scheduler.TuningOptions(
        num_measure_trials=63,  # Number of measurement trials
        measure_callbacks=[auto_scheduler.RecordToFile("matmul.json")],  # Log records
        verbose=2,  # Verbosity level
    )

    # Begin tuning process
    task.tune(tuning_options)

    # Apply the best schedule and build the function
    sch, args = task.apply_best("matmul.json")
    func = tvm.build(sch, args, target)

    dev = tvm.device("llvm", 0)
    # Create sample input arrays
    a_np = np.random.uniform(size=(M, K)).astype(DTYPE)
    b_np = np.random.uniform(size=(K, N)).astype(DTYPE)
    c_np = np.zeros((M, N), dtype=DTYPE)

    # Create TVM NDArray
    a_tvm = tvm.nd.array(a_np)
    b_tvm = tvm.nd.array(b_np)
    c_tvm = tvm.nd.array(c_np)

    evaluator = func.time_evaluator(func.entry_name, dev, number=5)
    exec_time = evaluator(a_tvm, b_tvm, c_tvm).mean
    print("Ansor execution time: %.3f ms" % (exec_time * 1e3))
    return exec_time


def get_baseline() -> float:
    # Find time of unchanged schedule
    env = create_schedule()
    func = tvm.build(
        env.schedule, env.static_args + [env.computed_arg], name="anything"
    )
    dev = tvm.device("llvm", 0)
    a = tvm.nd.array(numpy.random.rand(M, K).astype(DTYPE), dev)
    b = tvm.nd.array(numpy.random.rand(K, N).astype(DTYPE), dev)
    c = tvm.nd.array(numpy.zeros((M, N), dtype=DTYPE), dev)
    evaluator = func.time_evaluator(func.entry_name, dev, repeat=3)
    result = evaluator(a, b, c).mean
    return result


def get_blocking_baseline(bn=32, kfactor=4) -> float:
    env = create_schedule()
    # Apply blocking as described in https://tvm.apache.org/docs/v0.13.0/how_to/optimize_operators/opt_gemm.html
    C = env.computed_arg
    s = env.schedule
    mo, no, mi, ni = s[C].tile(C.op.axis[0], C.op.axis[1], bn, bn)
    (kaxis,) = s[C].op.reduce_axis
    ko, ki = s[C].split(kaxis, factor=kfactor)

    # Hoist reduction domain outside the blocking loop
    s[C].reorder(mo, no, ko, ki, mi, ni)

    func = tvm.build(
        env.schedule, env.static_args + [env.computed_arg], name="anything"
    )
    dev = tvm.device("llvm", 0)
    a = tvm.nd.array(numpy.random.rand(M, K).astype(DTYPE), dev)
    b = tvm.nd.array(numpy.random.rand(K, N).astype(DTYPE), dev)
    c = tvm.nd.array(numpy.zeros((M, N), dtype=DTYPE), dev)
    evaluator = func.time_evaluator(func.entry_name, dev, repeat=1)
    result = evaluator(a, b, c).mean
    return result


if __name__ == "__main__":
    baseline_score = get_baseline()
    ansor_baseline = get_ansor_baseline()
    print("Baseline:", baseline_score)

    tuner = RandomSearch(check_constraints=False)
    param = ScheduleParam(create_schedule, finish_schedule, 1, 5)
    for _ in range(RUNS):
        results.append([])
        tuner.tune(
            problem=Problem(
                "schedge",
                {"schedule": param},
                cost_function,
                [],
            ),
            budgets=[EvalBudget(ITERATIONS)],
        )

    best_block_schedule = float("inf")
    for bn_exp in range(1, 10):
        for kfactor_exp in range(1, 10):
            print(
                "Try default schedule with hyperparameters:", 2**bn_exp, 2**kfactor_exp
            )
            best_block_schedule = min(
                best_block_schedule, get_blocking_baseline(2**bn_exp, 2**kfactor_exp)
            )

    zipped_results = list(zip(*results))
    means = [np.mean(x) for x in zipped_results]
    mins = [np.min(x) for x in zipped_results]
    maxs = [np.max(x) for x in zipped_results]
    xs = range(len(zipped_results))
    plt.figure()
    plt.plot(xs, means, label="Random Search")
    plt.plot(xs, [ansor_baseline] * len(xs), label="Ansor Baseline")
    plt.fill_between(xs, mins, maxs, alpha=0.3)
    plt.plot(xs, [baseline_score] * len(xs), label="Baseline")
    plt.plot(
        xs,
        [best_block_schedule] * len(xs),
        label="Optimized Block Schedule",
    )
    plt.xlabel("function evaluations")
    plt.ylabel("cost")
    plt.yscale("log")
    plt.legend()
    plt.show()
