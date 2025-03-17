import os
import random
import sys
import uuid
from typing import Callable

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
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


def random_8_letter_string():
    return "".join([chr(ord("a") + random.randint(0, 27) % 26) for _ in range(8)])


class TacoOperation(Operation):
    """
    Represents a TACO (Tensor Algebra Compiler) operation that can be applied to tensors.
    """

    def __init__(
        self,
        name: str,
        function_call: Callable[
            [object, dict], list[Axis] | None
        ],  # Pass ScheduleEnv object, get back operation result
        params: dict[str, MethodParamType],
        return_type: MethodReturnType,
    ):
        super().__init__(
            name,
            lambda environment, args: function_call(environment["schedule_env"], args),
            params,
            return_type,
        )


def split(s, kwargs):
    first = random_8_letter_string()
    second = random_8_letter_string()
    s.split(kwargs["original"], first, second, kwargs["factor"])
    return [first, second]


SPLIT = TacoOperation(
    "split",
    split,
    {
        "original": AxisParam(consuming=True),
        "factor": ExpIntParam(2, min_exp=1, max_exp=8),
    },
    ReturnTypeAxesList(2),
)

REORDER = TacoOperation(
    "reorder",
    lambda s, kwargs: s.reorder(kwargs["order"]),
    {"order": AxisPoolPermutationParam()},
    ReturnTypeNone,
)


def fuse(s, kwargs):
    fused = random_8_letter_string()
    s.fuse(kwargs["original_first"], kwargs["original_second"], fused)
    return [fused]


FUSE = TacoOperation(
    "fuse",
    fuse,
    {
        "original_first": AxisParam(consuming=True),
        "original_second": AxisParam(consuming=True),
    },
    ReturnTypeAxesList(1),
)
