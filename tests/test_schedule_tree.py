import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from schedgehammer.schedule_type2 import AxisNode, OperationNode, ScheduleTree

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
