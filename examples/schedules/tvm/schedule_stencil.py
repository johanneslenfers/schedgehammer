import math

import numpy
import numpy as np
import tvm
from tvm import te

from schedgehammer.benchmark import benchmark
from tvm_api import REORDER, SPLIT, TILE
from schedgehammer.problem import Problem
from schedgehammer.schedules.schedule_type import ScheduleParam, ScheduleContext
from schedgehammer.tuner import EvalBudget
from schedgehammer.schedules.schedule_genetic_tuner import ScheduleGeneticTuner
from schedgehammer.schedules.schedule_random_search import ScheduleRandomSearch

I_SIZE = 512
STENCIL_RADIUS = 1
STENCIL_SIZE = STENCIL_RADIUS * 2 + 1
O_SIZE = I_SIZE - STENCIL_SIZE + 1

DTYPE = "float32"
ITERATIONS = 100  # If >63 limit ansors iterations else it will crash
RUNS = 3


def calculate_stencil(input_matrix, kernel):
    output = np.zeros((O_SIZE, O_SIZE))

    # Perform the convolution operation
    for i in range(O_SIZE):
        for j in range(O_SIZE):
            region = input_matrix[i:i + STENCIL_SIZE, j:j + STENCIL_SIZE]
            output[i, j] = np.sum(region * kernel)

    return output


def create_schedule() -> ScheduleContext:
    k = te.reduce_axis((-STENCIL_RADIUS, STENCIL_RADIUS), "k")
    j = te.reduce_axis((-STENCIL_RADIUS, STENCIL_RADIUS), "j")
    A = te.placeholder((I_SIZE, I_SIZE), name="A")
    S = te.placeholder((STENCIL_SIZE, STENCIL_SIZE), name="S")
    C = te.compute((O_SIZE, O_SIZE), lambda m, n: te.sum(te.sum(A[m + k + STENCIL_RADIUS, n + j + STENCIL_RADIUS] * S[k, j], axis=k), axis=j), name="O")

    # Default schedule
    s = te.create_schedule(C.op)

    return ScheduleContext(
        C.op.axis + C.op.axis,
        {
            'schedule': s,
            'tensor': C,
            'alltensors': [A, S, C],
        }
    )

a_np = numpy.random.rand(I_SIZE, I_SIZE).astype(DTYPE)
s_np = numpy.random.rand(STENCIL_SIZE, STENCIL_SIZE).astype(DTYPE)
correct_result = calculate_stencil(a_np, s_np)

def cost_function(config):
    dev = tvm.device("llvm", 0)

    # Random generated tensor for testing
    a = tvm.nd.array(a_np, dev)
    s = tvm.nd.array(s_np, dev)
    c = tvm.nd.array(numpy.zeros((O_SIZE, O_SIZE), dtype=DTYPE), dev)

    func: tvm.module.Module = config["schedule"]
    evaluator = func.time_evaluator(func.entry_name, dev, repeat=3, number=3)

    result = evaluator(a, s, c).mean
    if not np.allclose(c.numpy(), correct_result):
        return math.inf

    return result


def finish_schedule(ctx: ScheduleContext):
    return tvm.build(
        ctx.environment['schedule'],
        ctx.environment['alltensors'],
        name="anything",
    )


"""
def get_ansor_results():
    @auto_scheduler.register_workload
    def create_task_func():
        k = te.reduce_axis((0, K), "k")
        A = te.placeholder((I_SIZE, K), name="A")
        B = te.placeholder((K, N), name="B")
        C = te.compute((I_SIZE, N), lambda m, n: te.sum(A[m, k] * B[k, n], axis=k), name="C")
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
        a_np = np.random.uniform(size=(I_SIZE, K)).astype(DTYPE)
        b_np = np.random.uniform(size=(K, N)).astype(DTYPE)
        c_np = np.zeros((I_SIZE, N), dtype=DTYPE)

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
"""

def get_baseline() -> float:
    # Find time of unchanged schedule
    env = create_schedule().environment
    func = tvm.build(
        env['schedule'], env['alltensors'], name="anything"
    )
    dev = tvm.device("llvm", 0)
    a = tvm.nd.array(a_np, dev)
    s = tvm.nd.array(s_np, dev)
    c = tvm.nd.array(numpy.zeros((O_SIZE, O_SIZE), dtype=DTYPE), dev)
    evaluator = func.time_evaluator(func.entry_name, dev, repeat=3)
    result = evaluator(a, s, c).mean
    return result


if __name__ == "__main__":
    param = ScheduleParam(
        create_schedule,
        finish_schedule,
        2,
        10,
        api_description=[TILE, SPLIT, REORDER],
    )
    problem = Problem(
        "schedge",
        {"schedule": param},
        cost_function,
        [],
        init_solver=False
    )
    benchmark(
        problem,
        [EvalBudget(ITERATIONS)],
        {
            'Genetic Tuner': ScheduleGeneticTuner(),
            'Random Search': ScheduleRandomSearch(),
        },
        f"results/schedule/stencil",
        5,
        True,
    )
