import os
from ctypes import c_double, c_void_p, cdll
from dataclasses import dataclass

from schedgehammer.param_types import ExpIntParam
from schedgehammer.schedule_type import AxisParam, Operation, ScheduleTree

# Load TACO library
lib = cdll.LoadLibrary(os.path.join(os.path.dirname(__file__), "taco_api.so"))
lib.create_schedule.restype = c_void_p
lib.cost_function.restype = c_double
lib.get_computed_tensor.restype = c_void_p
lib.get_static_tensor.restype = c_void_p
lib.get_original_axes.restype = c_void_p
lib.get_stmt.restype = c_void_p


class TacoObject:
    pointer: c_void_p


@dataclass
class TacoAxis(TacoObject):
    pass


@dataclass
class TacoTensor(TacoObject):
    pass


@dataclass
class TacoSchedule(TacoObject):
    pass


def create_schedule_tree() -> ScheduleTree:
    env = lib.create_schedule()
    tree = ScheduleTree(
        schedule=TacoSchedule(lib.get_stmt(env)),
        computed_tensor=TacoTensor(lib.get_computed_tensor(env)),
        static_tensors=[TacoTensor(lib.get_static_tensor(env, i)) for i in range(2)],
    )
    for i in range(3):
        tree.add_original_axis(TacoAxis(lib.get_original_axes(env, i)))
    return tree


def finish_schedule(tree: ScheduleTree):
    lib.finish_schedule(tree.schedule.pointer)
    return tree.schedule
