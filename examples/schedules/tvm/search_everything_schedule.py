import itertools
import time
from typing import List

import opentuner
from opentuner import MeasurementInterface, ConfigurationManipulator, IntegerParameter, SwitchParameter, \
    PermutationParameter, Result
from opentuner.search.manipulator import PowerOfTwoParameter

import schedgehammer.param_types
from matrix_multiplication import mm_problem, mm_schedule
from tvm_api import TILE, SPLIT, REORDER
from schedgehammer.param_types import Param
from schedgehammer.schedules.schedule_type import Operation, AxisParam, AxisPoolPermutationParam, ScheduleParam, \
    ReturnTypeAxesList


class FunctionsTuner(MeasurementInterface):
    schedule_param: ScheduleParam
    operations: List[Operation]

    def __init__(self, args, schedule_param: ScheduleParam, operations: List[Operation]):
        super().__init__(args)
        self.schedule_param = schedule_param
        self.operations = operations

    def manipulator(self):
        manipulator = ConfigurationManipulator()

        axes_amount = 3

        for i in range(len(self.operations)):
            operation = self.operations[i]
            for param_name, param in operation.params.items():
                full_name = f"{i}-{param_name}"
                if isinstance(param, AxisParam):
                    manipulator.add_parameter(SwitchParameter(full_name, axes_amount))
                    if param.consuming:
                        axes_amount -= 1
                elif isinstance(param, AxisPoolPermutationParam):
                    manipulator.add_parameter(PermutationParameter(full_name, list(range(axes_amount))))
                elif isinstance(param, Param):
                    if isinstance(param, schedgehammer.param_types.ExpIntParam):
                        assert param.base == 2
                        manipulator.add_parameter(PowerOfTwoParameter(full_name, param.base ** param.min_exp, param.base ** param.min_exp))
                    else:
                        raise NotImplementedError
                else:
                    raise NotImplementedError("Param type is stupid.")

            if isinstance(operation.return_type, ReturnTypeAxesList):
                axes_amount += operation.return_type.axes_amount
        return manipulator

    def run(self, desired_result, input, limit):
        env = self.schedule_param.create_schedule()

        axes = env.axes

        for i in range(len(self.operations)):
            cfg = desired_result.configuration.data
            operation = self.operations[i]
            func_params = {}
            for param_name, param in operation.params.items():
                full_name = f"{i}-{param_name}"
                if isinstance(param, AxisParam):
                    func_params[param_name] = axes[cfg[full_name]]
                    if param.consuming:
                        axes.pop(cfg[full_name])
                elif isinstance(param, AxisPoolPermutationParam):
                    func_params[param_name] = [axes[cfg[full_name][i]] for i in range(len(cfg[full_name]))]
                elif isinstance(param, Param):
                    if isinstance(param, schedgehammer.param_types.ExpIntParam):
                        func_params[param_name] = cfg[full_name]
                    else:
                        raise NotImplementedError
                else:
                    raise NotImplementedError("Param type is stupid.")
            new_axes = operation.function_call(env.environment, func_params)
            if new_axes is not None:
                axes += new_axes

        schedule = self.schedule_param.finish_schedule(env)

        return Result(time=mm_problem.cost_function({'schedule': schedule}))

def main():
    operations = [TILE, SPLIT, REORDER]

    stuff = itertools.chain.from_iterable([
        itertools.product(operations, repeat=n) for n in range(5)
    ])

    #for chosen_functions in stuff:
    argparser = opentuner.default_argparser()
    args = argparser.parse_args()
    FunctionsTuner.main(args, mm_schedule, [TILE, SPLIT, REORDER])


if __name__ == "__main__":
    main()