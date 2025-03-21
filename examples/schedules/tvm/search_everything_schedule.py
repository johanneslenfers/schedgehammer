import itertools
import os
from multiprocessing.pool import Pool
from typing import List

import opentuner
from opentuner import MeasurementInterface, ConfigurationManipulator, IntegerParameter, SwitchParameter, \
    PermutationParameter, Result
from opentuner.search.manipulator import PowerOfTwoParameter
from opentuner.tuningrunmain import TuningRunMain

import schedgehammer.param_types
from matrix_multiplication import mm_problem, mm_schedule
from tvm_api import TILE, SPLIT, REORDER
from schedgehammer.param_types import Param
from schedgehammer.schedules.schedule_type import Operation, AxisParam, AxisPoolPermutationParam, ScheduleParam, \
    ReturnTypeAxesList
import multiprocessing as mp

def process_evaluation(problem, config, result_queue: mp.Queue):
    result_queue.put(problem.cost_function(config))

def run_in_process(problem, config, timeout):
    result_queue = mp.Queue()
    process = mp.Process(
        target=process_evaluation, args=(problem, config, result_queue)
    )

    try:
        process.start()
        process.join(timeout=timeout)
    except KeyboardInterrupt:
        process.terminate()
        exit()

    if process.is_alive():
        process.terminate()  # Terminate the process
        process.join()  # Clean up the terminated process
        return Result(state='TIMEOUT', time=float('inf'))
    else:
        # Get the result from the queue
        return Result(time=result_queue.get())


class FunctionsTuner(MeasurementInterface):
    schedule_param: ScheduleParam
    operations: List[Operation]
    name: str
    best_score = float('inf')

    def __init__(self, args, schedule_param: ScheduleParam, operations: List[Operation]):
        super().__init__(args)
        self.schedule_param = schedule_param
        self.operations = operations
        self.name = "run_" + "-".join([operation.name for operation in operations])

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
                        manipulator.add_parameter(PowerOfTwoParameter(full_name, param.base ** param.min_exp, param.base ** param.max_exp))
                    else:
                        raise NotImplementedError
                else:
                    raise NotImplementedError

            if isinstance(operation.return_type, ReturnTypeAxesList):
                axes_amount += operation.return_type.axes_amount
        return manipulator

    def run(self, desired_result, input, limit):
        env = self.schedule_param.create_schedule()

        axes = env.axes.copy()
        cfg = desired_result.configuration.data
        for i in range(len(self.operations)):
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
                    raise NotImplementedError
            new_axes = operation.function_call(env.environment, func_params)
            if new_axes is not None:
                axes += new_axes

        schedule = self.schedule_param.finish_schedule(env)

        result = run_in_process(mm_problem, {'schedule': schedule}, 30)

        if result.time < self.best_score:
            self.best_score = result.time

        return result

    def save_final_config(self, config):
        print('RESULT', self.name, self.best_score, config.data)

class NoDaemonProcess(mp.Process):
    @property
    def daemon(self):
        return False

    @daemon.setter
    def daemon(self, value):
        pass


class NoDaemonContext(type(mp.get_context())):
    Process = NoDaemonProcess

# We sub-class multiprocessing.pool.Pool instead of multiprocessing.Pool
# because the latter is only a wrapper function, not a proper class.
class NestablePool(mp.pool.Pool):
    def __init__(self, *args, **kwargs):
        kwargs['context'] = NoDaemonContext()
        super(NestablePool, self).__init__(*args, **kwargs)


possible_operations = [TILE, SPLIT, REORDER]

def run_opentuner_with_operations(operation_indices: List[int]):
    operations = [possible_operations[i] for i in operation_indices]
    name = "run_" + "-".join([operation.name for operation in operations])
    argparser = opentuner.default_argparser()
    args = argparser.parse_args(
        ['--stop-after', '60', '--label', name, '--database', f"sqlite:///data/opentuner_{name}.db", '--quiet']
    )
    tuner = FunctionsTuner(args, mm_schedule, operations)
    tuning_run_main = TuningRunMain(tuner, args)
    tuning_run_main.main()

def main():
    stuff = list(itertools.chain.from_iterable([
        itertools.product(range(len(possible_operations)), repeat=n) for n in range(5)
    ]))

    with NestablePool(os.cpu_count() // 2) as pool:
        pool.map(run_opentuner_with_operations, stuff)


if __name__ == "__main__":
    main()