# Only needed since this is in the same repo as schedgehammer.
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from evaulate_schedule_language import evaluate_problem_for_schedule_language

##############################################################
from taco_api import FUSE, REORDER, SPLIT

import taco_bindings
from schedgehammer.schedules.schedule_type import ScheduleContext

DEFAULT_RUNS = 5
DEFAULT_ITERATIONS = 100


def create_creation_function(problem_name: str):
    def create_schedule() -> ScheduleContext:
        """Create a TACO schedule for matrix-vector multiplication"""
        s = taco_bindings.ScheduleEnv(problem_name)

        initial_axes = s.get_initial_axes()
        return ScheduleContext(
            initial_axes,  # The initial axes from the statement
            {
                "schedule_env": s,  # Store the ScheduleEnv object
            },
        )

    return create_schedule


def finish_schedule(ctx: ScheduleContext):
    """Finalize the schedule"""
    # For TACO, we simply return the ScheduleEnv object
    return ctx.environment["schedule_env"]


def cost_function(config):
    s = config["schedule"]

    try:
        exec_time_ms = s.execute()  # This returns time in milliseconds
    except Exception as e:
        print(f"Error during execution: {e}")
        exec_time_ms = 10000000

    result = exec_time_ms / 1000.0

    print("COST:", result)
    return result


def evaluate_schedule(
    problem_name: str, runs=DEFAULT_RUNS, iterations=DEFAULT_ITERATIONS
):
    evaluate_problem_for_schedule_language(
        problem_name,
        "taco",
        create_creation_function(problem_name),
        cost_function,
        finish_schedule,
        [SPLIT, FUSE, REORDER],
        runs=runs,
        iterations=iterations,
    )


def full_benchmark():
    evaluate_schedule("spmv")
    # evaluate_schedule("mttkrp")
    # evaluate_schedule("sddmm")


if __name__ == "__main__":
    full_benchmark()
