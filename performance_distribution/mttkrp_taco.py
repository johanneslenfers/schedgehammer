# Adapter to provide mttkrp functions for taco performance_distribution.py
import taco_bindings
from examples.schedules.taco.taco_api_operations import FUSE, REORDER, SPLIT
from schedgehammer.schedules.schedule_type import ScheduleContext


def create_mttkrp_schedule() -> ScheduleContext:
    """Create a TACO schedule for MTTKRP operation."""
    # Create a TACO ScheduleEnv with the "mttkrp_dense" example
    s = taco_bindings.ScheduleEnv("mttkrp_dense")

    # Get initial axes from the statement
    initial_axes = s.get_initial_axes()
    return ScheduleContext(
        initial_axes,  # The initial axes from the statement
        {
            "schedule_env": s,  # Store the ScheduleEnv object
        },
    )


def mttkrp_cost_function(config) -> float:
    """Cost function for MTTKRP schedule evaluation."""
    s = config["schedule"]

    # Time the execution
    try:
        exec_time_ms = s.execute()  # This returns time in milliseconds
    except Exception as e:
        print(f"Error during execution: {e}")
        exec_time_ms = 10000000

    # Convert to seconds (to match TVM's timing)
    result = exec_time_ms / 1000.0

    return result

