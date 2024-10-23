from dataclasses import dataclass
from typing import Callable

from param_types import Param, ParamValue


@dataclass
class Problem:
    params: dict[str, Param]
    cost_function: Callable[[dict[str, ParamValue]], float]

    # Do we want to construct this with dict[str, Param], or List[Param], and build dict[str, Param] from there?
