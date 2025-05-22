from __future__ import annotations

import os
import random
import sys
import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Generic, TypeVar

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from schedgehammer.param_types import Param, ParamValue, T

Schedule = TypeVar("Schedule")
Axis = TypeVar("Axis")
Tensor = TypeVar("Tensor")
Environment = TypeVar("Environment")
CompiledSchedule = TypeVar("CompiledSchedule")


@dataclass
class AxisParam:
    consuming: bool  # Wether or not the axis gets removed from the pool
    force_inner: bool = False  # Wether or not it has to be the innermost axis


@dataclass
class AxisPoolPermutationParam:
    consuming: bool = False


MethodParamType = AxisParam | AxisPoolPermutationParam | Param


@dataclass
class ReturnTypeAxesList:
    axes_amount: int


class ReturnTypeNone:
    pass


MethodReturnType = ReturnTypeAxesList | ReturnTypeNone


@dataclass
class Operation(Generic[Axis, Tensor]):
    name: str
    function_call: Callable[
        [Environment, dict[str, ParamValue | Axis | list[Axis]]], list[Axis] | None
    ]  # Pass tensor, get back operation to call with the generated args
    params: dict[str, MethodParamType]
    return_type: MethodReturnType

    def preconditions_met(self, schedule_tree: SchedulePlanningTree):
        axes_needed = len(
            list(filter(lambda p: isinstance(p, AxisParam), self.params.values()))
        )
        axes_got = len(schedule_tree.unconsumed_axes)
        return axes_got >= axes_needed

    def apply_random_on_tree(self, schedule_tree: SchedulePlanningTree):
        operation_node = OperationNode(self, {}, {}, [])
        non_consumed_axes_nodes = []
        generated_axes_nodes = []

        for param_name, param_type in self.params.items():
            if isinstance(param_type, Param):
                operation_node.parameters[param_name] = param_type.choose_random()
            elif isinstance(param_type, AxisPoolPermutationParam):
                # AxisPoolPermutationParam uses all available axes.
                operation_node.input_axes[param_name] = (
                    schedule_tree.unconsumed_axes.copy()
                )
                random.shuffle(operation_node.input_axes[param_name])

                if param_type.consuming:
                    schedule_tree.unconsumed_axes = []
                else:
                    for axis_node in schedule_tree.unconsumed_axes:
                        non_consumed_axes_nodes.append(AxisNode(axis_node.id))
            elif isinstance(param_type, AxisParam):
                if param_type.force_inner:
                    param_val = max(schedule_tree.unconsumed_axes, key=lambda x: x.id)
                else:
                    param_val: AxisNode = random.choice(schedule_tree.unconsumed_axes)

                operation_node.input_axes[param_name] = param_val
                if param_type.consuming:
                    schedule_tree.unconsumed_axes.remove(param_val)
                else:
                    non_consumed_axes_nodes.append(AxisNode(param_val.id))

            else:
                raise NotImplementedError("unsupported parameter type")

        if isinstance(self.return_type, ReturnTypeAxesList):
            for _ in range(self.return_type.axes_amount):
                generated_axes_nodes.append(AxisNode(schedule_tree.next_axis_id))
                schedule_tree.next_axis_id += 1

        operation_node.output_axes = non_consumed_axes_nodes + generated_axes_nodes
        schedule_tree.unconsumed_axes += generated_axes_nodes

        for input_axis in operation_node.input_axes.values():
            if isinstance(input_axis, AxisNode):
                input_axis.processed_in = operation_node
            else:
                for axis_node in input_axis:
                    axis_node.processed_in = operation_node

        schedule_tree.operations.append(operation_node)


@dataclass
class SchedulePlanningTree:
    """
    Rules of the schedule tree:
    - Each AxisNode can only be processed once. If the operation is not consuming,
    create a new axisNode with the same id outgoing from the operation.
    """

    operations: list[OperationNode]
    unconsumed_axes: list[AxisNode] = field(default_factory=lambda: [])
    next_axis_id: int = 0

    def __init__(self, initial_axes_amount: int):
        self.operations = []
        self.initial_axes_amount = initial_axes_amount
        self.unconsumed_axes = [AxisNode(i) for i in range(initial_axes_amount)]
        self.next_axis_id = self.initial_axes_amount

    def visualize(self):
        """
        Can be called during debugging
        """
        import matplotlib.pyplot as plt
        import networkx as nx

        g = nx.Graph()

        def axis_node_name(node):
            return f"{node.id} | {id(node)}"

        def operation_node_name(node):
            return f"{node.operation.name} | {id(node)}"

        def node_operation(node):
            if isinstance(node, AxisNode):
                if node.processed_in:
                    g.add_edge(
                        axis_node_name(node),
                        operation_node_name(node.processed_in),
                    )
            else:
                for axis in node.output_axes:
                    g.add_edge(operation_node_name(node), axis_node_name(axis))

        self._do_inorder_traversal(node_operation)
        nx.draw(g, with_labels=True)
        plt.show()

    def _do_inorder_traversal(self, node_operation: Callable[[Node], None]):
        """
        Helper method doing in-order traversal of the tree and applying a custom
        operation on each node.
        """
        visited = set()

        def traverse(node: Node):
            if node is None or id(node) in visited:
                return
            visited.add(id(node))
            node_operation(node)
            for child in node.get_children():
                traverse(child)

        for op in self.operations:
            traverse(op)

    def __eq__(self, other):
        return id(self) == id(other)

    def __str__(self):
        result = ["--- Schedule Tree ---"]
        for node in self.operations:
            result.append(str(node))
        result.append("--- End of Schedule Tree ---")
        return "\n".join(result)

    def randomly_tweak_primitive_params(self) -> None:
        """
        Go through all operations and give the primitive params new random values.
        """
        for operation_node in self.operations:
            for param_name, param in operation_node.operation.params.items():
                if isinstance(param, Param):
                    operation_node.parameters[param_name] = param.choose_random()

    def delete_starting_with_operation(self, operation_index: int):
        while len(self.operations) > operation_index:
            operation = self.operations.pop()
            for axis in operation.input_axes.values():
                axis.processed_in = None
        del operation

    def randomly_merge_with_other_schedule(
        self,
        other_schedule: SchedulePlanningTree,
        max_length: int,
    ):
        """
        1. Pick any operation from this tree
        2. Remove it and all its children
        3. Reapply the schedule tree
        4. Pick a starting point in the topological order of the other tree
        5. Iteratively apply the operations from the other tree
        """
        random_node = random.choice(self.operations)

        self.delete_starting_with_operation(self.operations.index(random_node))

        to_be_added = other_schedule.operations
        start_index = random.randint(0, len(to_be_added) - 1)
        to_be_added = to_be_added[start_index:]
        for node in to_be_added:
            if len(self.operations) > max_length:
                break
            node.operation.apply_random_on_tree(self)


class Node(ABC):
    @abstractmethod
    def get_children(self) -> list[Node]:
        raise NotImplementedError


@dataclass
class AxisNode(Node):
    id: int
    processed_in: OperationNode | None = None

    def get_children(self) -> list[Node]:
        return [self.processed_in] if self.processed_in else []

    def __str__(self):
        return str(self.id)


@dataclass
class OperationNode(Node):
    operation: Operation
    parameters: dict[str, ParamValue]
    input_axes: dict[str, AxisNode | list[AxisNode]]
    output_axes: list[AxisNode]

    def get_children(self) -> list[Node]:
        return self.output_axes

    def __str__(self):
        return f"[{self.operation.name}|{'-'.join([str(axis) if isinstance(axis, AxisNode) else 'all' for axis in self.input_axes.values()])}|{'-'.join([str(axis) for axis in self.output_axes])}]"


@dataclass
class ScheduleContext:
    axes: list[Any]
    environment: Any


class ScheduleParam(Param[Any], Generic[CompiledSchedule]):
    create_schedule: Callable[[], ScheduleContext]
    finish_schedule: Callable[[ScheduleContext], CompiledSchedule]
    min_length: int
    max_length: int
    api_description: list[Operation]
    initial_axes_amount: int
    first_operation_blacklist: list[Operation] = field(default_factory=list)
    last_generated_tree: SchedulePlanningTree | None = None

    def __init__(
        self,
        create_schedule: Callable[[], ScheduleContext],
        finish_schedule: Callable[[ScheduleContext], CompiledSchedule],
        min_length: int,
        max_length: int,
        api_description: list[Operation],
        first_operation_blacklist: list[Operation] = None,
    ) -> None:
        self.create_schedule = create_schedule
        self.finish_schedule = finish_schedule
        self.min_length = min_length
        self.max_length = max_length
        self.api_description = api_description
        self.first_operation_blacklist = first_operation_blacklist or []
        # Execute create_schedule once in order to get amount of initial axes.
        self.initial_axes_amount = len(self.create_schedule().axes)

    def create_random_schedule(self) -> SchedulePlanningTree:
        while True:
            schedule_tree = SchedulePlanningTree(self.initial_axes_amount)
            desired_length = random.randint(self.min_length, self.max_length)

            while len(schedule_tree.operations) < desired_length:
                method_candidates = list(
                    filter(
                        lambda method: method.preconditions_met(schedule_tree),
                        self.api_description,
                    )
                )

                # Apply blacklist constraint for first operation
                if (
                    len(schedule_tree.operations) == 0
                    and self.first_operation_blacklist
                ):
                    method_candidates = list(
                        filter(
                            lambda method: method not in self.first_operation_blacklist,
                            method_candidates,
                        )
                    )

                if len(method_candidates) == 0:
                    break
                chosen_method: Operation = random.choice(method_candidates)
                chosen_method.apply_random_on_tree(schedule_tree)

            else:  # If inner while exited normally
                return schedule_tree

    def choose_random(self, current_value=None):
        while True:
            try:
                self.last_generated_tree = self.create_random_schedule()
                return self.last_generated_tree
            except Exception as e:
                traceback.print_exc()
                print(f"\033[93mFailed to create random schedule bc {e}\033[0m")

    def translate_for_evaluation(self, schedule_tree: SchedulePlanningTree) -> T:
        # print("Translate for evaluation:")
        # print(schedule_tree)
        schedule_context = self.create_schedule()
        # Dict to keep track of axis.
        axes = {i: axis for i, axis in enumerate(schedule_context.axes)}

        for i_operation, operation in enumerate(schedule_tree.operations):
            translated_input_axes: dict = {}

            amount_non_consumed_axes = 0

            for param_name, param in operation.input_axes.items():
                if isinstance(param, AxisNode):
                    translated_input_axes[param_name] = axes[param.id]
                    if not operation.operation.params[param_name].consuming:
                        amount_non_consumed_axes += 1
                else:
                    # AxisPoolPermutation
                    translated_input_axes[param_name] = [
                        axes[axis_node.id] for axis_node in param
                    ]
                    if not operation.operation.params[param_name].consuming:
                        amount_non_consumed_axes += len(
                            translated_input_axes[param_name]
                        )

            try:
                return_value = operation.operation.function_call(
                    schedule_context.environment,
                    translated_input_axes | operation.parameters,
                )
            except Exception as e:
                print(
                    f"Exception during operation {operation.operation.name} ({i_operation}): ",
                    e,
                )
                return None

            if return_value is not None:
                corresponding_axes = operation.output_axes[-len(return_value) :]
                for i in range(len(return_value)):
                    axes[corresponding_axes[i].id] = return_value[i]

            return self.finish_schedule(schedule_context)
