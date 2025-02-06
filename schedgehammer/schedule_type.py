from __future__ import annotations

import copy
import os
import random
import sys
import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Generic, Type, TypeVar

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from schedgehammer.param_types import Param, ParamValue

Schedule = TypeVar("Schedule")
Axis = TypeVar("Axis")
Tensor = TypeVar("Tensor")
CompiledSchedule = TypeVar("CompiledSchedule")


@dataclass
class AxisParam:
    consuming: bool  # Wether or not the axis gets removed from the pool
    force_inner: bool = False  # Wether or not it has to be the innermost axis


@dataclass
class AxisPoolPermutationParam:
    pass


MethodParamType = AxisParam | AxisPoolPermutationParam | Param


@dataclass
class Operation(Generic[Axis, Tensor]):
    name: str
    compiler_operation: Callable[
        [Tensor, list, dict], list[Axis]
    ]  # Pass tensor, get back operation to call with the generated args
    params: dict[str, MethodParamType]

    def apply_random_on_tree(self, schedule_tree: ScheduleTree) -> ScheduleTree:
        operation_node = OperationNode(self, {}, {}, [])
        non_consumed_axes_nodes = []
        generated_axes_nodes = []
        compiler_args = {}
        all_axes: list[AxisNode] = schedule_tree.get_leave_axes()
        for param_name, param_type in self.params.items():
            if isinstance(param_type, Param):
                param_val = param_type.choose_random()
                operation_node.args[param_name] = param_val
            elif isinstance(param_type, AxisPoolPermutationParam):
                operation_node.input_axes[param_name] = all_axes.copy()
                random.shuffle(all_axes)
                for axis_node in all_axes:
                    copied_axis = AxisNode(axis_node.id, axis_node.axis, None)
                    non_consumed_axes_nodes.append(copied_axis)
                param_val: list[Axis] = [axis.axis for axis in all_axes]
            elif isinstance(param_type, AxisParam):
                if param_type.force_inner:
                    param_val = max(all_axes, key=lambda x: x.id)
                    all_axes.remove(param_val)
                else:
                    param_val: AxisNode = all_axes.pop(random.randrange(len(all_axes)))
                operation_node.input_axes[param_name] = param_val
                if not param_type.consuming:
                    copied_axis = AxisNode(param_val.id, param_val.axis, None)
                    non_consumed_axes_nodes.append(copied_axis)
                param_val = param_val.axis
            compiler_args[param_name] = param_val
        # Operation payload is ready, now call it
        tensor = schedule_tree.schedule[schedule_tree.computed_tensor]
        if "" in compiler_args.keys():
            # Not a kwarg, but a positional arg
            new_axes = self.compiler_operation(
                tensor,
                compiler_args[""]
                if isinstance(compiler_args[""], list)
                else [compiler_args[""]],
                {},
            )
        else:
            new_axes = self.compiler_operation(tensor, [], compiler_args)
        # Create operation node and new axis nodes for the generated axes and put them in the tree
        for axis in new_axes if new_axes else []:
            generated_axes_nodes.append(AxisNode(schedule_tree.max_id + 1, axis, None))
            schedule_tree.max_id += 1
        operation_node.output_axes = non_consumed_axes_nodes + generated_axes_nodes
        for input_axis in operation_node.input_axes.values():
            # last step because operation might fail before this,
            # in that case we don't want to change the tree
            if isinstance(input_axis, AxisNode):
                input_axis.processed_in = operation_node
            else:
                for axis_node in input_axis:
                    axis_node.processed_in = operation_node


@dataclass
class ScheduleTree(Generic[Schedule, Axis, Tensor]):
    """
    Rules of the schedule tree:
    - Each AxisNode can only be processed once. If the operation is not consuming,
    create a new axisNode with the same id outgoing from the operation.
    """

    schedule: Schedule
    computed_tensor: Tensor
    static_tensors: list[Tensor]
    original_axes: list[AxisNode] = field(default_factory=lambda: [])
    max_id: int = 0
    meta: list[str] = field(default_factory=lambda: [])  # For debugging

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

    def reapply_schedule(
        self,
        fresh_schedule: Schedule,
        fresh_tensor: Tensor,
        fresh_static_tensors: list[Tensor],
        fresh_axes: list[Axis],
    ):
        """
        After modifying the tree, reapply the schedule.
        """
        self.static_tensors = fresh_static_tensors
        self.schedule = fresh_schedule
        self.computed_tensor = fresh_tensor
        tensor = self.schedule[self.computed_tensor]
        for i, axis in enumerate(fresh_axes):
            self.original_axes[i].axis = axis
        topological_order = self.get_topological_order()
        for node in topological_order:
            """
            Assumptions we can make at any step in the loop:
            - TvmAxes of the Parent nodes are already updated
            - For the output nodes of each operation node the leftmost nodes will be the
            non-consumed axes nodes.
            """
            compiler_args = node.args.copy()
            for arg_name, arg_val in node.input_axes.items():
                compiler_args[arg_name] = (
                    arg_val.axis  # The axis of the node
                    if isinstance(arg_val, AxisNode)  # If it is a single axis
                    else [axis_node.axis for axis_node in arg_val]  # Else the axis for
                    # each node in the list
                )
            if "" in compiler_args.keys():
                # Not a kwarg, but a positional arg
                try:
                    new_axes = (
                        node.operation.compiler_operation(
                            tensor,
                            compiler_args[""]
                            if isinstance(compiler_args[""], list)
                            else [compiler_args[""]],
                            {},
                        )
                        or []
                    )
                except Exception as e:
                    print(self.meta)
                    print(traceback.format_exc())
            else:
                try:
                    new_axes = (
                        node.operation.compiler_operation(tensor, [], compiler_args)
                        or []
                    )
                except Exception as e:
                    print(self.meta)
                    print(traceback.format_exc())
            parents_by_id = {}
            for parent in node.input_axes.values():
                if isinstance(parent, AxisNode):
                    parents_by_id[parent.id] = parent
                else:
                    for parent_axis_node in parent:
                        parents_by_id[parent_axis_node.id] = parent_axis_node

            new_axes_start_index = len(node.output_axes) - len(new_axes)
            for nonconsumed_axis_node in node.output_axes[:new_axes_start_index]:
                nonconsumed_axis_node.axis = parents_by_id[
                    nonconsumed_axis_node.id
                ].axis
            for i, generated_axis_node in enumerate(
                node.output_axes[new_axes_start_index:]
            ):
                generated_axis_node.axis = new_axes[i]

    def add_original_axis(self, axis: Axis):
        self.original_axes.append(AxisNode(self.max_id + 1, axis, None))
        self.max_id += 1

    def _do_bfs(self, node_operation: Callable[[Node], bool], start_node: Node = None):
        """
        Helper method doing bfs on the tree and applying a custom operation on each node.
        The node operation shall return True if the bfs should be ended.
        """
        queue = [start_node] if start_node else self.original_axes.copy()
        visited = set()

        while queue:
            node = queue.pop(0)
            if id(node) in visited:
                continue
            visited.add(id(node))
            if node_operation(node):
                return
            queue.extend(node.get_children())

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

        for axis in self.original_axes:
            traverse(axis)

    def get_leave_axes(self) -> list[AxisNode]:
        """
        Make in-order traversal of the tree and return just the leave Axes.
        """
        leave_axes = []

        def node_operation(node: Node):
            if isinstance(node, AxisNode) and node.processed_in is None:
                leave_axes.append(node)

        self._do_inorder_traversal(node_operation)
        assert not any(axes.processed_in for axes in leave_axes)
        return leave_axes

    def __len__(self):
        """
        The len of a schedule is the number of operations.
        """
        operation_nodes = []

        def node_operation(node: Node):
            if isinstance(node, OperationNode):
                operation_nodes.append(node)

        self._do_inorder_traversal(node_operation)
        return len(operation_nodes)

    def __eq__(self, other):
        return id(self) == id(other)

    def __str__(self):
        result = ["--- Schedule Tree ---"]
        nodes = self.get_topological_order(operations_only=False)
        for node in nodes:
            result.append(str(node))
        result.append("--- End of Schedule Tree ---")
        return "\n".join(result)

    def get_topological_order(self, operations_only=True) -> list[Node]:
        def sortUtil(node, visited, stack):
            visited.add(id(node))

            for element in node.get_children():
                if id(element) not in visited:
                    sortUtil(element, visited, stack)
            if isinstance(node, OperationNode) or not operations_only:
                stack.insert(0, node)

        visited = set()

        stack = []

        for element in self.original_axes:
            sortUtil(element, visited, stack)

        return stack

    def get_innermost_axis(self) -> AxisNode:
        """
        Get Axis Node with highest id
        """
        return max(self.get_leave_axes(), key=lambda x: x.id)

    def randomly_tweak_primitive_params(self) -> None:
        """
        Go through all operations and give the primitive params new random values.
        """
        for operation_node in self.get_topological_order():
            for param_name, param in operation_node.operation.params.items():
                if isinstance(param, Param):
                    operation_node.args[param_name] = param.choose_random()

    def randomly_merge_with_other_schedule(
        self,
        other_schedule: ScheduleTree,
        fresh_schedule: Schedule,
        fresh_tensor: Tensor,
        fresh_static_tensors: list[Tensor],
        fresh_axes: list[Axis],
        max_legnth: int,
    ):
        """
        1. Pick any operation from this tree
        2. Remove it and all its children
        3. Reapply the schedule tree
        4. Pick a starting point in the topological order of the other tree
        5. Iteratively apply the operations from the other tree
        """
        topological_order = self.get_topological_order()
        random_node = random.choice(topological_order)

        def node_operaton(node):
            if isinstance(node, AxisNode):
                return
            for input in node.input_axes.values():
                if isinstance(input, AxisNode):
                    input.processed_in = None
                else:
                    for axis_node in input:
                        axis_node.processed_in = None

        self._do_bfs(node_operaton, start_node=random_node)
        # TODO: Continue here, this causes bugs
        self.reapply_schedule(
            fresh_schedule, fresh_tensor, fresh_static_tensors, fresh_axes
        )
        to_be_added = other_schedule.get_topological_order()
        start_index = random.randint(0, len(topological_order) - 1)
        to_be_added = to_be_added[start_index:]
        for node in to_be_added:
            if len(self) > max_legnth:
                break
            node.operation.apply_random_on_tree(self)


class Node(ABC):
    @abstractmethod
    def get_children(self) -> list[Node]:
        raise NotImplementedError


@dataclass
class AxisNode(Node, Generic[Axis]):
    id: int
    axis: Axis
    processed_in: OperationNode | None = None

    def get_children(self) -> list[Node]:
        return [self.processed_in] if self.processed_in else []

    def __str__(self):
        return self.axis.var.name


@dataclass
class OperationNode:
    operation: Operation
    args: dict[str, ParamValue]
    input_axes: dict[str, AxisNode | list[AxisNode]]
    output_axes: list[AxisNode]

    def get_children(self) -> list[Node]:
        return self.output_axes

    def __str__(self):
        return f"[{self.operation.name}|{'-'.join([str(axis) if isinstance(axis, AxisNode) else 'all' for axis in self.input_axes.values()])}|{'-'.join([str(axis) for axis in self.output_axes])}]"


@dataclass
class ScheduleParam(Param[Any], Generic[CompiledSchedule]):
    create_schedule: Callable[[None], ScheduleTree]
    finish_schedule: Callable[[ScheduleTree], CompiledSchedule]
    min_length: int
    max_length: int
    api_description: list[Operation]
    terminating_methods: list[Operation]
    last_generated_tree: ScheduleTree | None = None

    def try_appending_method(
        self,
        schedule_tree: ScheduleTree,
        method_candidates: list[Operation],
    ) -> tuple[ScheduleTree, list[Operation], Operation | None]:
        """
        Tries to append one of the method candidates to the schedule.
        Returns:
            - Updated schedule
            - Updated method candidates
            - Chosen method type
        """
        method = method_candidates.pop(random.randrange(len(method_candidates)))
        try:
            method.apply_random_on_tree(schedule_tree)
            if len(schedule_tree) < self.min_length:
                method_candidates = [
                    method
                    for method in self.api_description
                    if method not in self.terminating_methods
                ]
            else:
                method_candidates = self.api_description.copy()
            return schedule_tree, method_candidates, method
        except Exception:
            print(f"\033[93mCouldn't apply {method.name}\033[0m")
        return schedule_tree, method_candidates, None

    def create_random_schedule(self) -> ScheduleTree:
        while True:
            schedule_tree = self.create_schedule()
            desired_length = random.randint(self.min_length, self.max_length)
            method_candidates = [
                method
                for method in self.api_description
                if method not in self.terminating_methods
            ]
            while len(schedule_tree) < desired_length:
                schedule_tree, method_candidates, method = self.try_appending_method(
                    schedule_tree, method_candidates
                )
                if method in self.terminating_methods:
                    break
                if not method_candidates:
                    break
            if len(schedule_tree) >= self.min_length:
                break
        return schedule_tree

    def choose_random(self, current_value=None):
        while True:
            try:
                self.last_generated_tree = self.create_random_schedule()
                return self.finish_schedule(self.last_generated_tree)
            except Exception as e:
                print(f"\033[93mFailed to create random schedule bc {e}\033[0m")
