import taco_bindings
from schedgehammer.benchmark import benchmark
from schedgehammer.problem import Problem
from schedgehammer.schedules.schedule_genetic_tuner import ScheduleGeneticTuner
from schedgehammer.schedules.schedule_random_search import ScheduleRandomSearch
from schedgehammer.schedules.schedule_type import ScheduleContext, ScheduleParam
from schedgehammer.tuner import EvalBudget, Tuner
from taco_api_operations import FUSE, SPLIT, REORDER

# Parameters for the example
def create_schedule() -> ScheduleContext:
    """Create a TACO schedule for matrix-vector multiplication"""
    # Create a TACO ScheduleEnv with the "sddmm" example
    s = taco_bindings.ScheduleEnv("sddmm")

    # Get initial axes from the statement
    initial_axes = s.get_initial_axes()
    return ScheduleContext(
        initial_axes,  # The initial axes from the statement
        {
            "schedule_env": s,  # Store the ScheduleEnv object
        },
    )

def finish_schedule(ctx: ScheduleContext):
    """Finalize the schedule"""
    # For TACO, we simply return the ScheduleEnv object
    return ctx.environment["schedule_env"]


class TacoProblem(Problem):

    def __init__(self):
        super().__init__(
            "taco-sddmm",
            {"schedule": ScheduleParam(
                create_schedule,
                finish_schedule,
                2,
                8,
                api_description=[SPLIT, FUSE, REORDER],
                first_operation_blacklist=[FUSE, REORDER],
            )},
            [],
            init_solver=False
        )


    def cost_function(self, config):
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

if __name__ == "__main__":
    benchmark(
        TacoProblem,
        [EvalBudget(100)],
        {
            "RandomSearch": ScheduleRandomSearch(),
            "GeneticTuner": ScheduleGeneticTuner(),
        },
        output_path="results/taco/sddmm",
        repetitions=64,
        export_raw_data=True,
        parallel=64
    )
