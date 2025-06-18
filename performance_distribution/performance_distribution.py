import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

from examples.schedules.taco.mttkrp_dense import create_schedule, finish_schedule
from examples.schedules.taco.taco_api_operations import REORDER, SPLIT
from schedgehammer.schedules.schedule_type import ScheduleParam

NUM_SCHEDULES = 10
VARIANTS_PER_SCHEDULE = 10

if os.path.exists("/cloud/wwu1"):
    print("Running on palma II")
    RESULTS_PATH = "/scratch/tmp/sspehr/performance_distribution_results.json"
else:
    RESULTS_PATH = "performance_distribution_results.json"


def cost_function(config):
    # Debug: print the keys in the config dictionary

    s = config["schedule"]

    # Time the execution
    try:
        exec_time_ms = s.execute()  # This returns time in milliseconds
    except Exception as e:
        print(f"Error during execution: {e}")
        exec_time_ms = 10000000

    # Convert to seconds (to match TVM's timing)
    result = exec_time_ms / 1000.0

    # If the reported time is 0, use the Python-measured time
    # if result == 0:
    #     result = end - start

    # Record the best result so far

    return result


def evaluate_single_schedule(i):
    print(i)
    schedule_instance_costs = []
    param = ScheduleParam(
        create_schedule,
        finish_schedule,
        2,
        10,
        api_description=[SPLIT, REORDER],
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
