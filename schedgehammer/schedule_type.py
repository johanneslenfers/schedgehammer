from __future__ import annotations

import copy
import os
import random
import sys
import traceback
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, ClassVar, Generator, Type

import tvm
from tvm import te
from tvm.te import Tensor
from tvm.tir.expr import IterVar as TvmAxis

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from schedgehammer.param_types import ExpIntParam, Param, ParamValue


@dataclass
class TvmAxisParam:
    consuming: bool  # Wether or not the axis gets removed from the pool
    force_inner: bool = False  # Wether or not it has to be the innermost axis


@dataclass
class TvmAxisPoolPermutationParam:
    pass


MethodParamType = TvmAxisParam | TvmAxisPoolPermutationParam | Param


@dataclass
class Operation:
    name: str
    tvm_operation: Callable[
        [Tensor, list, dict], list[TvmAxis]
    ]  # Pass tensor, get back operation to call with the generated args
    params: dict[str, MethodParamType]

    def apply_random_on_tree(self, schedule_tree: ScheduleTree) -> ScheduleTree:
        operation_node = OperationNode(self, {}, {}, [])
        non_consumed_axes_nodes = []
        generated_axes_nodes = []
        tvm_args = {}
        all_axes: list[AxisNode] = schedule_tree.get_leave_axes()
        for param_name, param_type in self.params.items():
            if isinstance(param_type, Param):
                param_val = param_type.choose_random()
                operation_node.args[param_name] = param_val
            elif isinstance(param_type, TvmAxisPoolPermutationParam):
                operation_node.input_axes[param_name] = all_axes.copy()
                random.shuffle(all_axes)
                for axis_node in all_axes:
                    copied_axis = AxisNode(axis_node.id, axis_node.axis, None)
                    non_consumed_axes_nodes.append(copied_axis)
                param_val: list[TvmAxis] = [axis.axis for axis in all_axes]
            elif isinstance(param_type, TvmAxisParam):
                param_val: AxisNode = all_axes.pop(random.randrange(len(all_axes)))
                operation_node.input_axes[param_name] = param_val
                if not param_type.consuming:
                    copied_axis = AxisNode(param_val.id, param_val.axis, None)
                    non_consumed_axes_nodes.append(copied_axis)
                param_val = param_val.axis
            tvm_args[param_name] = param_val
        # Operation payload is ready, now call it
        tensor = schedule_tree.tvm_schedule[schedule_tree.computed_tensor]
        if "" in tvm_args.keys():
            # Not a kwarg, but a positional arg
            new_axes = self.tvm_operation(
                tensor,
                tvm_args[""] if isinstance(tvm_args[""], list) else [tvm_args[""]],
                {},
            )
        else:
            new_axes = self.tvm_operation(tensor, [], tvm_args)
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


TILE = Operation(
    "tile",
    lambda t, args, kwargs: t.tile(*args, **kwargs),
    {
        "x_parent": TvmAxisParam(consuming=True),
        "y_parent": TvmAxisParam(consuming=True),
        "x_factor": ExpIntParam(2, min_exp=1, max_exp=8),
        "y_factor": ExpIntParam(2, min_exp=1, max_exp=8),
    },
)
SPLIT = Operation(
    "split",
    lambda t, args, kwargs: t.split(*args, **kwargs),
    {
        "parent": TvmAxisParam(consuming=True),
        "factor": ExpIntParam(2, min_exp=1, max_exp=8),
    },
)
REORDER = Operation(
    "reorder",
    lambda t, args, kwargs: t.reorder(*args, **kwargs),
    {"": TvmAxisPoolPermutationParam()},
)
# VECTORIZE = Operation(
#     "vectorize",
#     lambda t: t.vectorize,
#     {"": TvmAxisParam(consuming=True, force_inner=True)},
# )


@dataclass
class ScheduleTree:
    """
    Rules of the schedule tree:
    - Each AxisNode can only be processed once. If the operation is not consuming,
    create a new axisNode with the same id outgoing from the operation.
    """

    tvm_schedule: tvm.schedule.Schedule
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
        fresh_tvm_schedule: tvm.schedule.Schedule,
        fresh_tensor: Tensor,
        fresh_static_tensors: list[Tensor],
        fresh_axes: list[TvmAxis],
    ):
        """
        After modifying the tree, reapply the schedule.
        """
        self.static_tensors = fresh_static_tensors
        self.tvm_schedule = fresh_tvm_schedule
        self.computed_tensor = fresh_tensor
        tensor = self.tvm_schedule[self.computed_tensor]
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
            tvm_args = node.args.copy()
            for arg_name, arg_val in node.input_axes.items():
                tvm_args[arg_name] = (
                    arg_val.axis  # The tvm axis of the node
                    if isinstance(arg_val, AxisNode)  # If it is a single axis
                    else [axis_node.axis for axis_node in arg_val]  # Else the axis for
                    # each node in the list
                )
            if "" in tvm_args.keys():
                # Not a kwarg, but a positional arg
                try:
                    new_axes = (
                        node.operation.tvm_operation(
                            tensor,
                            tvm_args[""]
                            if isinstance(tvm_args[""], list)
                            else [tvm_args[""]],
                            {},
                        )
                        or []
                    )
                except Exception as e:
                    print(self.meta)
                    print(traceback.format_exc())
            else:
                try:
                    new_axes = node.operation.tvm_operation(tensor, [], tvm_args) or []
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

    def add_original_axis(self, axis: TvmAxis):
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

    def randomly_merge_with_other_schedule(
        self,
        other_schedule: ScheduleTree,
        fresh_tvm_schedule: tvm.schedule.Schedule,
        fresh_tensor: Tensor,
        fresh_static_tensors: list[Tensor],
        fresh_axes: list[TvmAxis],
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
            fresh_tvm_schedule, fresh_tensor, fresh_static_tensors, fresh_axes
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
class AxisNode(Node):
    id: int
    axis: TvmAxis
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
class ScheduleParam(Param[Any]):
    create_schedule: Callable[[None], ScheduleTree]
    finish_schedule: Callable[[ScheduleTree], Any]
    cost_function: Callable[[ScheduleTree], float]
    min_length: int
    max_length: int
    api_description: list[Operation] = field(
        default_factory=lambda: [TILE, SPLIT, REORDER]
    )
    terminating_methods: list[Operation] = field(
        default_factory=lambda: []  # Vectorize
    )
    use_genetic_algorithm_internally: bool = False
    population_size: int = 5
    elitism_share: float = 0.3
    reproduction_share: float = 0.5
    crossover_prob: float = 0.5
    additional_mutation_prob: float = 0.1
    local_mutation: bool = False
    current_population: list[tuple[ScheduleTree, float]] = field(default_factory=list)
    genetic_algorithm_callback: Callable[[ScheduleTree, float], None] | None = None

    def __post_init__(self, *args, **kwargs):
        if self.use_genetic_algorithm_internally:
            for _ in range(self.population_size):
                schedule_tree = self.create_random_schedule()
                cost = self.cost_function(
                    {"schedule": self.finish_schedule(schedule_tree)}
                )
                self.current_population.append((schedule_tree, cost))
        self.current_population = sorted(self.current_population, key=lambda x: x[1])
        print([schedule[1] for schedule in self.current_population])

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
            print(traceback.format_exc())
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
        if not self.use_genetic_algorithm_internally:
            return self.finish_schedule(self.create_random_schedule())
        else:
            elitism_size = int(self.population_size * self.elitism_share)
            reproduction_size = int(self.population_size * self.reproduction_share)
            new_population = self.current_population[:elitism_size]
            for _ in range(self.population_size - elitism_size):
                if random.random() < self.crossover_prob:
                    parent_one = copy.deepcopy(
                        random.choice(self.current_population[:reproduction_size])[0]
                    )
                    parent_two = copy.deepcopy(
                        random.choice(
                            [
                                element
                                for element in self.current_population[
                                    :reproduction_size
                                ]
                                if element[0] != parent_one
                            ]
                        )[0]
                    )
                    if random.random() < 0.5:
                        parent_one, parent_two = parent_two, parent_one
                    parent_one.meta.append("CROSSOVER")
                    fresh_tree = self.create_schedule()
                    parent_one.randomly_merge_with_other_schedule(
                        parent_two,
                        fresh_tree.tvm_schedule,
                        fresh_tree.computed_tensor,
                        fresh_tree.static_tensors,
                        [axis.axis for axis in fresh_tree.original_axes],
                        self.max_length,
                    )
                    while random.random() < self.additional_mutation_prob:
                        method_candidates = self.api_description.copy()
                        while method_candidates:
                            parent_one, _, method = self.try_appending_method(
                                parent_one, method_candidates
                            )
                            parent_one.meta.append("CROSSOVER_MUTATION")
                            if method in self.terminating_methods:
                                break

                    cost = self.cost_function(
                        {"schedule": self.finish_schedule(parent_one)}
                    )
                    new_population.append((parent_one, cost))
                else:
                    random_schedule = self.create_random_schedule()
                    cost = self.cost_function(
                        {"schedule": self.finish_schedule(random_schedule)}
                    )
                    new_population.append((random_schedule, cost))
            if self.genetic_algorithm_callback:
                self.genetic_algorithm_callback(*new_population[-1])
            self.current_population = sorted(new_population, key=lambda x: x[1])
            print([schedule[1] for schedule in self.current_population])
            return self.finish_schedule(self.current_population[0][0])


# if __name__ == "__main__":
#     M = 512
#     K = 512
#     N = 512

#     def create_schedule() -> ScheduleTree:
#         k = te.reduce_axis((0, K), "k")
#         A = te.placeholder((M, K), name="A")
#         B = te.placeholder((K, N), name="B")
#         C = te.compute((M, N), lambda m, n: te.sum(A[m, k] * B[k, n], axis=k), name="C")

#         # Default schedule
#         s = te.create_schedule(C.op)
#         schedule_tree = ScheduleTree(s, C)
#         schedule_tree.add_original_axis(C.op.axis[0])
#         schedule_tree.add_original_axis(C.op.axis[1])
#         schedule_tree.add_original_axis(C.op.reduce_axis[0])
#         return schedule_tree

#     schedule_tree = create_schedule()
#     print(schedule_tree)
#     REORDER.apply_random_on_tree(schedule_tree)
#     SPLIT.apply_random_on_tree(schedule_tree)
#     TILE.apply_random_on_tree(schedule_tree)
#     print(schedule_tree)
#     print(schedule_tree)
#     print(schedule_tree)

#     param = ScheduleParam(create_schedule, None, None, 2, 5)
#     for i in range(10):
#         print("Random tree", i)
#         st = param.create_random_schedule()
#         print(st)
#         print(
#             [
#                 node.operation.name
#                 for node in st.get_topological_order()
#                 if isinstance(node, OperationNode)
#             ]
#         )
#         print("Trying to reapply schedule saved in tree to new schedule")
#         fresh_tree = create_schedule()
#         st.reapply_schedule(
#             fresh_tree.tvm_schedule,
#             fresh_tree.computed_tensor,
#             [axis_node.axis for axis_node in fresh_tree.original_axes],
#         )
