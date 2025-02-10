import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from schedgehammer.param_types import ExpIntParam
from schedgehammer.schedule_type import AxisParam, AxisPoolPermutationParam, Operation

TILE = Operation(
    "tile",
    lambda tree, args, kwargs: tree.schedule[tree.computed_tensor].tile(
        *args, **kwargs
    ),
    {
        "x_parent": AxisParam(consuming=True),
        "y_parent": AxisParam(consuming=True),
        "x_factor": ExpIntParam(2, min_exp=1, max_exp=8),
        "y_factor": ExpIntParam(2, min_exp=1, max_exp=8),
    },
)
SPLIT = Operation(
    "split",
    lambda tree, args, kwargs: tree.schedule[tree.computed_tensor].split(
        *args, **kwargs
    ),
    {
        "parent": AxisParam(consuming=True),
        "factor": ExpIntParam(2, min_exp=1, max_exp=8),
    },
)
REORDER = Operation(
    "reorder",
    lambda tree, args, kwargs: tree.schedule[tree.computed_tensor].reorder(
        *args, **kwargs
    ),
    {"": AxisPoolPermutationParam()},
)

UNROLL = Operation(
    "unroll",
    lambda tree, args, kwargs: tree.schedule[tree.computed_tensor].unroll(
        *args, **kwargs
    ),
    {"": AxisParam(consuming=True, force_inner=True)},
)

PARALLEL = Operation(
    "parallel",
    lambda tree, args, kwargs: tree.schedule[tree.computed_tensor].parallel(
        *args, **kwargs
    ),
    {"": AxisParam(consuming=True, force_inner=True)},
)
