from abc import ABC, abstractmethod

import tvm
from tvm import auto_scheduler
from tvm.auto_scheduler import HardwareParams
from tvm.auto_scheduler.measure import PythonBasedMeasureCallback, ProgramRunner

from examples.schedules.tvm.tvm_api import TILE, SPLIT, REORDER
from schedgehammer.problem import Problem
from schedgehammer.schedules.schedule_type import ScheduleContext, ScheduleParam

class TVMScheduleProblem(Problem, ABC):

    def __init__(self, name: str):
        param = ScheduleParam(
            self.create_schedule,
            self.finish_schedule,
            2,
            10,
            api_description=[TILE, SPLIT, REORDER],
        )

        super().__init__(
            name,
            {"schedule": param},
            [],
            init_solver=False,
        )

    @abstractmethod
    def create_schedule(self) -> ScheduleContext:
        pass

    def finish_schedule(self, ctx: ScheduleContext):
        return tvm.build(
            ctx.environment["schedule"],
            ctx.environment["alltensors"],
            "llvm",
        )

    def get_ansor_results(self, iterations, runs):
        ansor_results = []

        @auto_scheduler.register_workload
        def create_task_func():
            s = self.create_schedule()
            return s.environment['alltensors']

        class StoreResultCallback(PythonBasedMeasureCallback):
            def callback(self, policy, inputs, results):
                for result in results[0:]:
                    ansor_results[-1].append(float(result.costs[0]))

        # Create the search task
        target = tvm.target.Target("llvm")
        for _ in range(runs):
            ansor_results.append([])
            task = auto_scheduler.SearchTask(
                func=create_task_func,
                target=target,
                hardware_params=HardwareParams(num_cores=1, target=target)
            )

            tuning_options = auto_scheduler.TuningOptions(
                num_measure_trials=iterations,
                measure_callbacks=[
                    auto_scheduler.RecordToFile("conv_2d.json"),
                    StoreResultCallback(),
                ],
                verbose=2,
            )

            # Begin tuning process
            task.tune(tuning_options)
        return ansor_results
