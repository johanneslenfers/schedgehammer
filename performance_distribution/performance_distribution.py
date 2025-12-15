import argparse
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

NAME = "mttkrp_tvm"

# Determine results path - use RESULTS_DIR environment variable if set, otherwise default location
RESULTS_DIR = os.environ.get("RESULTS_DIR", ".")
if os.path.exists("/cloud/wwu1"):
    print("Running on palma II")
    RESULTS_PATH = f"/scratch/tmp/sspehr/performance_distribution_results_{NAME}.json"
else:
    RESULTS_PATH = os.path.join(
        RESULTS_DIR, f"performance_distribution_results_{NAME}.json"
    )


def finish_schedule(ctx):
    return tvm.build(
        ctx.environment["schedule"],
        ctx.environment["alltensors"],
        "llvm --opt-level=0",
        name="anything",
    )


def evaluate_single_schedule(args):
    i, variants_per_schedule = args
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
    for _ in range(variants_per_schedule):
        cost = cost_function({"schedule": param.translate_for_evaluation(tree)})
        schedule_instance_costs.append(cost)
        tree.randomly_tweak_primitive_params()
    return schedule_instance_costs


def evaluate_performance_distribution(num_schedules, variants_per_schedule):
    num_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", multiprocessing.cpu_count()))
    print(f"Using {num_cpus} CPUs for multiprocessing.")
    print(
        f"Evaluating {num_schedules} schedules with {variants_per_schedule} variants each."
    )

    with ProcessPoolExecutor(max_workers=num_cpus) as executor:
        results = list(
            executor.map(
                evaluate_single_schedule,
                [(i, variants_per_schedule) for i in range(num_schedules)],
            )
        )

    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print(
        f"Saved {len(results)} schedules with {variants_per_schedule} variants each to {RESULTS_PATH}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run performance distribution analysis"
    )
    parser.add_argument(
        "--num-schedules",
        type=int,
        default=500,
        help="Number of schedules to evaluate (default: 500)",
    )
    parser.add_argument(
        "--variants-per-schedule",
        type=int,
        default=50,
        help="Number of variants per schedule (default: 50)",
    )
    args = parser.parse_args()

    evaluate_performance_distribution(args.num_schedules, args.variants_per_schedule)
