from dataclasses import dataclass
from typing import Callable

from schedgehammer.param_types import Param, ParamValue


@dataclass
class Problem:
    params: dict[str, Param]
    cost_function: Callable[[dict[str, ParamValue]], float]
