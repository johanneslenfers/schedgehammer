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
from schedgehammer.schedules.schedule_genetic_tuner import ScheduleGeneticTuner
from schedgehammer.schedules.schedule_random_search import ScheduleRandomSearch
from schedgehammer.schedules.schedule_type import ScheduleContext, ScheduleParam
from schedgehammer.tuner import EvalBudget, Tuner

M = 1024
K = 1024
N = 1024

DTYPE = "float32"
ITERATIONS = 200  # If >63 limit ansors iterations else it will crash
RUNS = 19

ansor_results = []
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


class StoreResultCallback(PythonBasedMeasureCallback):
    def callback(self, policy, inputs, results):
        for result in results[0:]:
            cost = float(result.costs[0])
            if not ansor_results[-1] or cost < ansor_results[-1][-1]:
                ansor_results[-1].append(cost)
            else:
                ansor_results[-1].append(ansor_results[-1][-1])


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
    dev = tvm.device("llvm", 0)

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
        name="anything",
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

        evaluator = func.time_evaluator(func.entry_name, dev, repeat=3, number=3)
        exec_time = evaluator(a_tvm, b_tvm, c_tvm).median
        print("Ansor execution time: %.3f ms" % (exec_time * 1e3))
        ansor_results[-1][-1] = (
            exec_time  # Make sure the real execution time is not higher than calculated by ansor
        )


def get_baseline() -> float:
    # Find time of unchanged schedule
    env = create_schedule().environment
    func = tvm.build(env["schedule"], env["alltensors"], name="anything")
    dev = tvm.device("llvm", 0)
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

    func = tvm.build(s, env["alltensors"], name="anything")
    dev = tvm.device("llvm", 0)
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

    benchmark(
        MMProblem,
        [EvalBudget(ITERATIONS)],
        {
            "genetic_tuner": ScheduleGeneticTuner(),
            "random_tuner": ScheduleRandomSearch(),
        },
        f"data/mm_schedule",
        RUNS,
        True
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
