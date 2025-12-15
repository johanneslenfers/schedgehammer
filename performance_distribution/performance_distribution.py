import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

import tvm
from api import REORDER, SPLIT, TILE
from mttkrp import create_mttkrp_schedule as create_schedule
from mttkrp import mttkrp_cost_function as cost_function

from schedgehammer.schedules.schedule_type import ScheduleParam

NUM_SCHEDULES = 500
VARIANTS_PER_SCHEDULE = 50
NAME = "mttkrp_tvm"

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


def evaluate_single_schedule(i):
    print(i)
    schedule_instance_costs = []
    param = ScheduleParam(
        create_schedule,
        finish_schedule,
        2,
        10,
        api_description=[SPLIT, REORDER, TILE],
    )
    tree = param.choose_random()
    for _ in range(VARIANTS_PER_SCHEDULE):
        cost = cost_function({"schedule": param.translate_for_evaluation(tree)})
        schedule_instance_costs.append(cost)
        tree.randomly_tweak_primitive_params()
    return schedule_instance_costs


def evaluate_performance_distribution():
    num_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", multiprocessing.cpu_count()))
    print(f"Using {num_cpus} CPUs for multiprocessing.")

    with ProcessPoolExecutor(max_workers=num_cpus) as executor:
        results = list(executor.map(evaluate_single_schedule, range(NUM_SCHEDULES)))

    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print(
        f"Saved {len(results)} schedules with {VARIANTS_PER_SCHEDULE} variants each to performance_distribution_results.json"
    )


if __name__ == "__main__":
    evaluate_performance_distribution()
