import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

import tvm
from api import PARALLEL, REORDER, SPLIT, TILE, UNROLL, VECTORIZE
from mm import K, M, N
from mm import create_mm_schedule as create_schedule
from mm import mm_cost_function as cost_function
from tvm import auto_scheduler, te
from tvm.auto_scheduler.measure import PythonBasedMeasureCallback

from schedgehammer.schedules.schedule_type import ScheduleContext, ScheduleParam

NUM_SCHEDULES = 20000
NUM_CHORES = 32
NAME = "beat_ansor"

if os.path.exists("/cloud/wwu1"):
    print("Running on palma II")
    RESULTS_PATH = f"/scratch/tmp/sspehr/performance_distribution_results_{NAME}.json"
else:
    RESULTS_PATH = f"performance_distribution_results_{NAME}.json"


def finish_schedule(ctx):
    return tvm.build(
        ctx.environment["schedule"],
        ctx.environment["alltensors"],
        "llvm --opt-level=0",
        name="anything",
    )


def get_ansor_mm_results():
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
                ansor_results.append(cost)

    # Create the search task
    target = tvm.target.Target("llvm")
    task = auto_scheduler.SearchTask(func=create_task_func, target=target)

    tuning_options = auto_scheduler.TuningOptions(
        num_measure_trials=63,
        measure_callbacks=[
            StoreResultCallback(),
        ],
        verbose=2,
    )

    # Begin tuning process
    task.tune(tuning_options)
    return ansor_results


def evaluate_batch(i):
    schedule_instance_costs = []
    param = ScheduleParam(
        create_schedule,
        finish_schedule,
        2,
        10,
        api_description=[SPLIT, REORDER, TILE, UNROLL, PARALLEL],
    )
    for j in range(NUM_SCHEDULES // NUM_CHORES):
        tree = param.choose_random()
        if j % 100 == 0:
            print(f"{j}|{i}")
        cost = cost_function({"schedule": param.translate_for_evaluation(tree)})
        schedule_instance_costs.append(cost)
    print(f"Finished {i}")
    return schedule_instance_costs


def evaluate_performance_distribution():
    ansor_results = get_ansor_mm_results()
    num_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", multiprocessing.cpu_count()))
    print(f"Using {num_cpus} CPUs for multiprocessing.")

    with ProcessPoolExecutor(max_workers=num_cpus) as executor:
        results = list(executor.map(evaluate_batch, range(NUM_CHORES)))
    results.append(ansor_results)
    print("Write out")
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    evaluate_performance_distribution()
