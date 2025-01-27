import json
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy
import numpy as np
import tvm
from matplotlib import pyplot as plt
from tvm import auto_scheduler, te
from tvm.auto_scheduler.measure import PythonBasedMeasureCallback

from schedgehammer.problem import Problem
from schedgehammer.random_search import RandomSearch
from schedgehammer.schedule_type import ScheduleParam, ScheduleTree
from schedgehammer.schedule_type_old import ScheduleEnvironment
from schedgehammer.schedule_type_old import ScheduleParam as ScheduleParamOld
from schedgehammer.tuner import EvalBudget

M = 512
K = 512
N = 512

DTYPE = "float32"
ITERATIONS = 15  # If >63 limit ansors iterations else it will crash
RUNS = 6

results_genetic = []
results_random = []
ansor_results = []


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


class StoreResultCallback(PythonBasedMeasureCallback):
    def callback(self, policy, inputs, results):
        for result in results[0:]:
            cost = float(result.costs[0])
            if not ansor_results[-1] or cost < ansor_results[-1][-1]:
                ansor_results[-1].append(cost)
            else:
                ansor_results[-1].append(ansor_results[-1][-1])


def create_schedule() -> ScheduleTree:
    k = te.reduce_axis((0, K), "k")
    A = te.placeholder((M, K), name="A")
    B = te.placeholder((K, N), name="B")
    C = te.compute((M, N), lambda m, n: te.sum(A[m, k] * B[k, n], axis=k), name="C")

    # Default schedule
    s = te.create_schedule(C.op)

    tree = ScheduleTree(
        tvm_schedule=s,
        computed_tensor=C,
        static_tensors=[A, B],
    )
    tree.add_original_axis(C.op.axis[0])
    tree.add_original_axis(C.op.axis[1])
    tree.add_original_axis(C.op.reduce_axis[0])
    return tree


def create_cost_function(result_list):
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
        if not result_list[-1] or result.mean < result_list[-1][-1]:
            result_list[-1].append(result.mean)
        else:
            result_list[-1].append(result_list[-1][-1])
        return result.mean

    return cost_function


def finish_schedule(tree: ScheduleTree):
    return tvm.build(
        tree.tvm_schedule, tree.static_tensors + [tree.computed_tensor], name="anything"
    )


def get_ansor_results():
    @auto_scheduler.register_workload
    def create_task_func():
        k = te.reduce_axis((0, K), "k")
        A = te.placeholder((M, K), name="A")
        B = te.placeholder((K, N), name="B")
        C = te.compute((M, N), lambda m, n: te.sum(A[m, k] * B[k, n], axis=k), name="C")
        return [A, B, C]

    # Create the search task
    target = tvm.target.Target("llvm")
    for _ in range(RUNS):
        ansor_results.append([])
        task = auto_scheduler.SearchTask(func=create_task_func, target=target)

        tuning_options = auto_scheduler.TuningOptions(
            num_measure_trials=ITERATIONS,
            measure_callbacks=[
                auto_scheduler.RecordToFile("matmul.json"),
                StoreResultCallback(),
            ],
            verbose=2,
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
        ansor_results[-1][-1] = (
            exec_time  # Make sure the real execution time is not higher than calculated by ansor
        )


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
    POPULATION_SIZE = 5
    ELITISM_SHARE = 0.3
    tuner = RandomSearch(check_constraints=False)
    all_genetically_generated_trees = []

    def genetic_param_callback(tree: ScheduleTree, cost: float):
        all_genetically_generated_trees.append(((str(tree), tree.meta), cost))

    for _ in range(RUNS):
        results_genetic.append([])
        param = ScheduleParam(
            create_schedule,
            finish_schedule,
            create_cost_function(results_genetic),
            2,
            10,
            population_size=POPULATION_SIZE,
            elitism_share=ELITISM_SHARE,
            use_genetic_algorithm_internally=True,
            genetic_algorithm_callback=genetic_param_callback,
        )
        tuner.tune(
            problem=Problem(
                "schedge",
                {"schedule": param},
                create_cost_function(results_genetic),
                [],
            ),
            budgets=[
                EvalBudget(
                    (ITERATIONS - POPULATION_SIZE)
                    / POPULATION_SIZE
                    * (1 - ELITISM_SHARE)
                )
            ],
        )
    all_genetically_generated_trees.sort(key=lambda x: x[1])
    with open("all_genetic_trees.json", "w") as file:
        file.write(json.dumps(all_genetically_generated_trees))
    tuner = RandomSearch(check_constraints=False)
    for _ in range(RUNS):
        results_random.append([])
        param = ScheduleParam(
            create_schedule,
            finish_schedule,
            create_cost_function(results_random),
            2,
            10,
            use_genetic_algorithm_internally=False,
        )
        tuner.tune(
            problem=Problem(
                "schedge",
                {"schedule": param},
                create_cost_function(results_random),
                [],
            ),
            budgets=[EvalBudget(ITERATIONS)],
        )

    # baseline_score = get_baseline()
    # print("Baseline:", baseline_score)
    # get_ansor_results()
    # best_block_schedule = float("inf")
    # for bn_exp in range(1, 10):
    #     for kfactor_exp in range(1, 10):
    #         print(
    #             "Try default schedule with hyperparameters:", 2**bn_exp, 2**kfactor_exp
    #         )
    #         best_block_schedule = min(
    #             best_block_schedule, get_blocking_baseline(2**bn_exp, 2**kfactor_exp)
    #         )

    plt.figure()
    plot_results_from_several_runs(results_genetic, "Genetic Search")
    plot_results_from_several_runs(results_random, "Random Search Old")
    # xs = plot_results_from_several_runs(ansor_results, "Ansor")
    # plt.plot(xs, [baseline_score] * len(xs), label="Baseline")
    # plt.plot(
    #     xs,
    #     [best_block_schedule] * len(xs),
    #     label="Optimized Block Schedule",
    # )
    plt.xlabel("function evaluations")
    plt.ylabel("cost")
    plt.yscale("log")
    plt.legend()
    plt.show()
