import taco_bindings
from schedgehammer.problem import Problem
from schedgehammer.schedules.schedule_random_search import ScheduleRandomSearch
from schedgehammer.schedules.schedule_type import ScheduleContext, ScheduleParam
from schedgehammer.tuner import EvalBudget, Tuner


def create_spmv_schedule() -> ScheduleContext:
    """Create a TACO schedule for matrix-vector multiplication"""
    s = taco_bindings.ScheduleEnv("spmv")

    initial_axes = s.get_initial_axes()
    return ScheduleContext(
        initial_axes,  # The initial axes from the statement
        {
            "schedule_env": s,  # Store the ScheduleEnv object
        },
    )


def spmv_cost_function(config):
    s = config["schedule"]

    try:
        exec_time_ms = s.execute()  # This returns time in milliseconds
    except Exception as e:
        print(f"Error during execution: {e}")
        exec_time_ms = 10000000

    result = exec_time_ms / 1000.0

    print("COST:", result)
    return result
