from typing import Callable

from schedgehammer.param_types import ExpIntParam
from schedgehammer.schedules.schedule_type import (
    Axis,
    AxisParam,
    AxisPoolPermutationParam,
    MethodParamType,
    MethodReturnType,
    Operation,
    ReturnTypeAxesList,
    ReturnTypeNone,
    Tensor,
)


class TvmOperation(Operation):
    def __init__(
        self,
        name: str,
        function_call: Callable[
            [Tensor, dict], list[Axis] | None
        ],  # Pass tensor, get back operation to call with the generated args
        params: dict[str, MethodParamType],
        return_type: MethodReturnType,
    ):
        super().__init__(
            name,
            lambda environment, args: function_call(
                environment["schedule"][environment["tensor"]], args
            ),
            params,
            return_type,
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
    ReturnTypeAxesList(4),
)
SPLIT = TvmOperation(
    "split",
    lambda t, kwargs: t.split(**kwargs),
    {
        "parent": AxisParam(consuming=True),
        "factor": ExpIntParam(2, min_exp=1, max_exp=8),
    },
    ReturnTypeAxesList(2),
)
REORDER = TvmOperation(
    "reorder",
    lambda t, args: t.reorder(*args["order"]),
    {"order": AxisPoolPermutationParam()},
    ReturnTypeNone,
)
