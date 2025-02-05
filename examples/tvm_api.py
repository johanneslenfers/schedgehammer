import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from schedgehammer.param_types import ExpIntParam
from schedgehammer.schedule_type import AxisParam, AxisPoolPermutationParam, Operation

TILE = Operation(
    "tile",
    lambda t, args, kwargs: t.tile(*args, **kwargs),
    {
        "x_parent": AxisParam(consuming=True),
        "y_parent": AxisParam(consuming=True),
        "x_factor": ExpIntParam(2, min_exp=1, max_exp=8),
        "y_factor": ExpIntParam(2, min_exp=1, max_exp=8),
    },
)
SPLIT = Operation(
    "split",
    lambda t, args, kwargs: t.split(*args, **kwargs),
    {
        "parent": AxisParam(consuming=True),
        "factor": ExpIntParam(2, min_exp=1, max_exp=8),
    },
)
REORDER = Operation(
    "reorder",
    lambda t, args, kwargs: t.reorder(*args, **kwargs),
    {"": AxisPoolPermutationParam()},
)

UNROLL = Operation(
    "unroll",
    lambda t, args, kwargs: t.unroll(*args, **kwargs),
    {"": AxisParam(consuming=True, force_inner=True)},
)

PARALLEL = Operation(
    "parallel",
    lambda t, args, kwargs: t.parallel(*args, **kwargs),
    {"": AxisParam(consuming=True, force_inner=True)},
)
# VECTORIZE = Operation(
#     "vectorize",
#     lambda t: t.vectorize,
#     {"": TvmAxisParam(consuming=True, force_inner=True)},
# )
