import argparse
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

from mttkrp_taco import create_mttkrp_schedule as create_schedule
from mttkrp_taco import mttkrp_cost_function as cost_function

from examples.schedules.taco.taco_api_operations import FUSE, REORDER, SPLIT
from schedgehammer.schedules.schedule_type import ScheduleParam

NAME = "mttkrp_taco"
# Explicitly ensure NAME is correct - this is the TACO script
print(f"[TACO SCRIPT] Starting - NAME is set to: {NAME}")
print(f"[TACO SCRIPT] This file is: {__file__}")
assert NAME == "mttkrp_taco", f"NAME must be 'mttkrp_taco' but got '{NAME}'"

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
    # For TACO, we simply return the ScheduleEnv object
    return ctx.environment["schedule_env"]


def evaluate_single_schedule(args):
    i, variants_per_schedule = args
    print(i)
    schedule_instance_costs = []
    param = ScheduleParam(
        create_schedule,
        finish_schedule,
        2,
        8,
        api_description=[SPLIT, FUSE, REORDER],
        first_operation_blacklist=[FUSE, REORDER],
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

    # Reconstruct path at runtime to ensure correct NAME
    # Double-check NAME is still correct
    assert NAME == "mttkrp_taco", (
        f"NAME changed! Expected 'mttkrp_taco' but got '{NAME}'"
    )
    results_dir = os.environ.get("RESULTS_DIR", ".")
    if os.path.exists("/cloud/wwu1"):
        results_path = (
            f"/scratch/tmp/sspehr/performance_distribution_results_{NAME}.json"
        )
    else:
        results_path = os.path.join(
            results_dir, f"performance_distribution_results_{NAME}.json"
        )

    # Final verification
    assert "mttkrp_taco" in results_path, (
        f"Path should contain 'mttkrp_taco' but got: {results_path}"
    )

    print(f"[TACO SCRIPT] About to save to: {results_path}")
    print(f"[TACO SCRIPT] NAME variable is: {NAME}")

    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(
        f"Saved {len(results)} schedules with {variants_per_schedule} variants each to {results_path}"
    )

    # Verify what was actually written
    if os.path.exists(results_path):
        print(f"[TACO SCRIPT] ✓ File exists at: {results_path}")
    else:
        print(f"[TACO SCRIPT] ✗ File NOT found at: {results_path}")


if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(
            description="Run performance distribution analysis for TACO"
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

        evaluate_performance_distribution(
            args.num_schedules, args.variants_per_schedule
        )
    except Exception as e:
        print(f"ERROR in performance_distribution_taco.py: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)
