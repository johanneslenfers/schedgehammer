import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from tvm import te

from schedgehammer.schedule_type import (
    REORDER,
    SPLIT,
    TILE,
    AxisNode,
    OperationNode,
    ScheduleParam,
    ScheduleTree,
)

s = ScheduleTree([], None, None)
ax1 = AxisNode(11, None, None)
ax2 = AxisNode(21, None, None)
ax3 = AxisNode(31, None, None)
ax1i = AxisNode(41, None, None)
ax1o = AxisNode(51, None, None)
ax99 = AxisNode(99, None, None)
split1 = OperationNode(None, {}, {}, [ax1i, ax1o])
ax1.processed_in = split1
ax2.processed_in = split1
ax1i_copy = AxisNode(42, None, None)
ax1o_copy = AxisNode(52, None, None)
ax2_copy = AxisNode(22, None, None)
ax3_copy = AxisNode(32, None, None)
reorder = OperationNode(None, {}, {}, [ax1i_copy, ax2_copy, ax1o_copy, ax3_copy])
ax1i.processed_in = reorder
ax2.processed_in = reorder
ax1o.processed_in = reorder
ax3.processed_in = reorder
ax1ii = AxisNode(61, None, None)
ax1io = AxisNode(71, None, None)
split2 = OperationNode(None, {}, {}, [ax1ii, ax1io])
ax1i_copy.processed_in = split2
s.original_axes = [ax1, ax2, ax3, ax99]


def test_tree_traversal():
    # Create a sample schedule tree
    assert [ax.id for ax in s.get_leave_axes()] == [61, 71, 22, 52, 32, 99]


def test_no_crash():
    M = 512
    K = 512
    N = 512

    def create_schedule() -> ScheduleTree:
        k = te.reduce_axis((0, K), "k")
        A = te.placeholder((M, K), name="A")
        B = te.placeholder((K, N), name="B")
        C = te.compute((M, N), lambda m, n: te.sum(A[m, k] * B[k, n], axis=k), name="C")

        # Default schedule
        s = te.create_schedule(C.op)
        schedule_tree = ScheduleTree(s, C, [A, B])
        schedule_tree.add_original_axis(C.op.axis[0])
        schedule_tree.add_original_axis(C.op.axis[1])
        schedule_tree.add_original_axis(C.op.reduce_axis[0])
        return schedule_tree

    param = ScheduleParam(create_schedule, None, None, 2, 5)
    for i in range(10):
        print("Random tree", i)
        st = param.create_random_schedule()
        print(st)
        print(
            [
                node.operation.name
                for node in st.get_topological_order()
                if isinstance(node, OperationNode)
            ]
        )
        print("Trying to reapply schedule saved in tree to new schedule")
        fresh_tree = create_schedule()
        st.reapply_schedule(
            fresh_tree.tvm_schedule,
            fresh_tree.computed_tensor,
            fresh_tree.static_tensors,
            [axis_node.axis for axis_node in fresh_tree.original_axes],
        )

    s1 = param.create_random_schedule()
    s2 = param.create_random_schedule()
    fresh_tree = create_schedule()
    s1.randomly_merge_with_other_schedule(
        s2,
        fresh_tree.tvm_schedule,
        fresh_tree.computed_tensor,
        fresh_tree.static_tensors,
        [axis_node.axis for axis_node in fresh_tree.original_axes],
        7,
    )
