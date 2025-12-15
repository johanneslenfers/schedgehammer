import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

from examples.schedules.taco.mttkrp_dense import create_schedule, finish_schedule
from examples.schedules.taco.taco_api_operations import REORDER, SPLIT
from schedgehammer.schedules.schedule_type import ScheduleParam

variants_per_schedule = int(sys.argv[1])
iteration = int(sys.argv[2])

RESULTS_PATH = "taco_performance_distribution_results.json"


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
    for _ in range(variants_per_schedule):
        cost = cost_function({"schedule": param.translate_for_evaluation(tree)})
        schedule_instance_costs.append(cost)
        tree.randomly_tweak_primitive_params()
    return schedule_instance_costs


if __name__ == "__main__":
    with open(RESULTS_PATH, "r") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            data = []
    new_costs = evaluate_single_schedule(iteration)
    data.append(new_costs)
    with open(RESULTS_PATH, "w") as f:
        json.dump(data, f, indent=2)
