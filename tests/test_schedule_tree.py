import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from tvm import te

from schedgehammer.schedules.schedule_type import (
    ScheduleParam,
    ScheduleContext,
)


def test_no_crash():
    M = 512
    K = 512
    N = 512

    def create_schedule() -> ScheduleContext:
        k = te.reduce_axis((0, K), "k")
        A = te.placeholder((M, K), name="A")
        B = te.placeholder((K, N), name="B")
        C = te.compute((M, N), lambda m, n: te.sum(A[m, k] * B[k, n], axis=k), name="C")

        # Default schedule
        s = te.create_schedule(C.op)

        return ScheduleContext(
            [C.op.axis[0], C.op.axis[1], C.op.reduce_axis[0]],
            {
                'schedule': s,
                'tensor': C,
                'alltensors': [A, B, C],
            }
        )

    param = ScheduleParam(create_schedule, None, None, 2, 5)
    for i in range(10):
        print("Random tree", i)
        st = param.create_random_schedule()
        print(st)
        print(
            [
                node.operation.name
                for node in st.operations
            ]
        )
        print("Trying to reapply schedule saved in tree to new schedule")
        fresh_tree = create_schedule()
        param.translate_for_evaluation(st)

    s1 = param.create_random_schedule()
    s2 = param.create_random_schedule()
    fresh_tree = create_schedule()
    s1.randomly_merge_with_other_schedule(
        s2,
        fresh_tree.compiler_schedule,
        fresh_tree.computed_tensor,
        fresh_tree.static_tensors,
        [axis_node.axis for axis_node in fresh_tree.original_axes],
        7,
    )
