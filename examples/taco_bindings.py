import os
import sys
from ctypes import c_double, c_void_p, cdll
from dataclasses import dataclass

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from schedgehammer.param_types import ExpIntParam
from schedgehammer.schedule_type import (
    AxisParam,
    AxisPoolPermutationParam,
    Operation,
    ScheduleTree,
)

# Load TACO library
lib = cdll.LoadLibrary(os.path.join(os.path.dirname(__file__), "taco_api.so"))
lib.create_schedule.restype = c_void_p
lib.cost_function.restype = c_double
lib.get_computed_tensor.restype = c_void_p
lib.get_static_tensor.restype = c_void_p
lib.get_original_axes.restype = c_void_p
lib.get_stmt.restype = c_void_p
lib.generate_new_axis.restype = c_void_p


@dataclass
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
    env_pointer: c_void_p


def create_schedule_tree() -> ScheduleTree:
    env = lib.create_schedule()
    tree = ScheduleTree(
        schedule=TacoSchedule(lib.get_stmt(env), env),
        computed_tensor=TacoTensor(lib.get_computed_tensor(env)),
        static_tensors=[TacoTensor(lib.get_static_tensor(env, i)) for i in range(2)],
    )
    for i in range(3):
        tree.add_original_axis(TacoAxis(lib.get_original_axes(env, i)))
    return tree


def finish_schedule(tree: ScheduleTree):
    lib.finish_schedule(
        tree.static_tensors[0].pointer,
        tree.static_tensors[1].pointer,
        tree.computed_tensor.pointer,
    )
    return tree


def cost_function(config: dict):
    tree = config["schedule"]
    cost = lib.cost_function(
        tree.static_tensors[0].pointer,
        tree.static_tensors[1].pointer,
        tree.computed_tensor.pointer,
    )
    return cost


def split(tree, args, kwargs):
    # Create TacoAxis objects to maintain the Python wrapper around the C++ pointers
    inner_ptr = lib.generate_new_axis()
    outer_ptr = lib.generate_new_axis()
    inner = TacoAxis(inner_ptr)
    outer = TacoAxis(outer_ptr)
    parent = kwargs["parent"].pointer
    split_factor = kwargs["split_factor"]
    lib.split(tree.schedule.pointer, parent, outer_ptr, inner_ptr, split_factor)
    return [inner, outer]


SPLIT = Operation(
    "split",
    split,
    {
        "parent": AxisParam(consuming=True),
        "split_factor": ExpIntParam(2, min_exp=1, max_exp=8),
    },
)


def reorder(tree, args, kwargs):
    axes = [axis.pointer for axis in kwargs["axes"]]
    count_axes = len(axes)
    lib.reorder(tree.schedule.pointer, axes, count_axes)
    return []


REORDER = Operation(
    "reorder",
    reorder,
    {"axes": AxisPoolPermutationParam()},
)

if __name__ == "__main__":
    tree = create_schedule_tree()
    print("Tree created")
    SPLIT.apply_random_on_tree(tree)
    finish_schedule(tree)
    print(tree)
    print(cost_function({"schedule": tree}))
