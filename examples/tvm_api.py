import os
import sys
from typing import Callable

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from schedgehammer.param_types import ExpIntParam
from schedgehammer.schedules.schedule_type import AxisParam, AxisPoolPermutationParam, Operation, ReturnTypeAxesList, \
    ReturnTypeNone, MethodReturnType, MethodParamType, Axis, Tensor


class TvmOperation(Operation):

    def __init__(self,
                name: str,
                function_call: Callable[
                    [Tensor, dict],
                    list[Axis] | None
                ],  # Pass tensor, get back operation to call with the generated args
                params: dict[str, MethodParamType],
                return_type: MethodReturnType,
                 ):
        super().__init__(
            name,
            lambda environment, args: function_call(
                environment['schedule'][environment['tensor']],
                args
            ),
            params,
            return_type
        )


TILE = TvmOperation(
    "tile",
    lambda t, kwargs: t.tile(**kwargs),
    {
        "x_parent": AxisParam(consuming=True),
        "y_parent": AxisParam(consuming=True),
        "x_factor": ExpIntParam(2, min_exp=1, max_exp=8),
        "y_factor": ExpIntParam(2, min_exp=1, max_exp=8),
    },
    ReturnTypeAxesList(4)
)
SPLIT = TvmOperation(
    "split",
    lambda t, kwargs: t.split(**kwargs),
    {
        "parent": AxisParam(consuming=True),
        "factor": ExpIntParam(2, min_exp=1, max_exp=8),
    },
    ReturnTypeAxesList(2)
)
REORDER = TvmOperation(
    "reorder",
    lambda t, args: t.reorder(*args["order"]),
    {"order": AxisPoolPermutationParam()},
    ReturnTypeNone
)

UNROLL = TvmOperation(
    "unroll",
    lambda t, args: t.unroll(args["axis"]),
    {"axis": AxisParam(consuming=True, force_inner=True)},
    ReturnTypeNone
)

PARALLEL = TvmOperation(
    "parallel",
    lambda t, args: t.parallel(args["axis"]),
    {"axis": AxisParam(consuming=True, force_inner=True)},
    ReturnTypeNone
)
# VECTORIZE = TvmOperation(
#     "vectorize",
#     lambda t, args: t.vectorize(args["axis"]),
#     {"axis": AxisParam(consuming=True, force_inner=True)},
# )
