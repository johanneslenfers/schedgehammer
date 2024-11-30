from __future__ import annotations

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, ClassVar

import tvm
from tvm.te import Tensor
from tvm.tir.expr import IterVar as TvmAxis

from schedgehammer.param_types import ExpIntParam, Param, ParamValue


@dataclass
class TvmAxisParam:
    consuming: bool  # Wether or not the axis gets removed from the pool


@dataclass
class TvmAxisPoolPermutationParam:
    pass


MethodParamType = TvmAxisParam | TvmAxisPoolPermutationParam | Param
MethodParam = TvmAxis | list[TvmAxis] | ParamValue
MethodReturnType = None | TvmAxis | list[TvmAxis]


@dataclass
class Method(ABC):
    """
    Abstract class for the methods that can be applied to a schedule.
    A child class of this abstract class represents one sort of method (e.g. tiling) and
    an instance of a child class represents a concrete application of that method.
    """

    # Attributes of method itself (ClassVars)
    name: ClassVar[str]
    param_types: ClassVar[dict[str, MethodParamType]]
    return_type: ClassVar[type[MethodReturnType]]
    # Attributes of concrete method application
    params: dict[str, MethodParam]

    @classmethod
    @abstractmethod
    def is_possible(
        cls, previous_steps: list[Method], current_axis_pool: list[TvmAxis]
    ) -> Method:
        """
        Checks if it is possible to meaningfully apply the method.
        IMPORTANT: Write down assumptions about hard and soft constraints in the docstring.
        """
        raise NotImplementedError

    @classmethod
    def get_random(cls, current_axis_pool: list[TvmAxis], tensor: Tensor) -> Method:
        params = {}
        for param_name, param_type in cls.param_types.items():
            if isinstance(param_type, TvmAxisParam):
                random_axis_index = random.randint(0, len(current_axis_pool) - 1)
                if param_type.consuming:
                    param_val = current_axis_pool.pop(random_axis_index)
                else:
                    param_val = current_axis_pool[random_axis_index]
            elif isinstance(param_type, Param):
                param_val = param_type.choose_random()
            elif isinstance(param_type, TvmAxisPoolPermutationParam):
                # List of axis
                random.shuffle(current_axis_pool)
                param_val = current_axis_pool.copy()
            else:
                raise NotImplementedError
            params[param_name] = param_val
        method = cls(params=params)
        current_axis_pool += method.apply(tensor)
        return method

    @abstractmethod
    def apply(self, tensor: Tensor) -> list[TvmAxis]:
        raise NotImplementedError


@dataclass
class Tile(Method):
    name = "tile"
    param_types = {
        "x_parent": TvmAxisParam(consuming=True),
        "y_parent": TvmAxisParam(consuming=True),
        "x_factor": ExpIntParam(2, min_exp=1, max_exp=8),
        "y_factor": ExpIntParam(2, min_exp=1, max_exp=8),
    }
    return_type = list[TvmAxis]

    @classmethod
    def is_possible(
        cls, previous_steps: list[Method], current_axis_pool: list[TvmAxis]
    ) -> bool:
        """
        Assumptions:
            - Tiling requires at least two axes to be applied.
        """
        num_axes_available = len(current_axis_pool)
        return num_axes_available >= 2

    def apply(self, tensor: Tensor) -> list[TvmAxis]:
        return list(tensor.tile(**self.params))


@dataclass
class Split(Method):
    name = "split"
    param_types = {
        "parent": TvmAxisParam(consuming=True),
        "factor": ExpIntParam(2, min_exp=1, max_exp=8),
    }
    return_type = list[TvmAxis]

    @classmethod
    def is_possible(
        cls, previous_steps: list[Method], current_axis_pool: list[TvmAxis]
    ) -> bool:
        """
        Assumptions:
            - Splitting requires at least one axis to be present
        """
        num_axes_available = len(current_axis_pool)
        return num_axes_available >= 1

    def apply(self, tensor):
        return list(tensor.split(**self.params))


@dataclass
class Reorder(Method):
    name = "reorder"
    param_types = {
        "*axes": TvmAxisPoolPermutationParam(),
    }
    return_type = None

    @classmethod
    def is_possible(
        cls, previous_steps: list[Method], current_axis_pool: list[TvmAxis]
    ) -> bool:
        """
        Assumptions:
            - Reordering requires at least two axes to be meaningful
            - Previous step wasn't reordering
        """
        num_axes_available = len(current_axis_pool)
        if num_axes_available < 2:
            return False
        if previous_steps and previous_steps[-1].name == cls.name:
            return False
        return True

    def apply(self, tensor):
        tensor.reorder(*self.params["*axes"])
        return []


@dataclass
class ScheduleEnvironment:
    schedule: tvm.schedule.Schedule
    static_args: list[Tensor]
    computed_arg: Tensor
    axis_pool: list[TvmAxis]


@dataclass
class ScheduleParam(Param[list[Method]]):
    create_schedule: Callable[[None], ScheduleEnvironment]
    finish_schedule: Callable[[ScheduleEnvironment], tvm.module.Module]
    min_length: int
    max_length: int
    api_description: list[Method] = field(
        default_factory=lambda: [Tile, Split, Reorder]
    )
    successful_random_generations: int = 0
    failed_random_generations: int = 0

    def _add_next_method(
        self,
        method_candidates: list[Method],
        schedule: list[Method],
        env: ScheduleEnvironment,
    ) -> tuple[list[Method], list[Method]]:
        """
        Adds next method to the schedule and returns updated schedule and updated set
        of method candidates for the next step.
        """
        method = method_candidates.pop(random.randrange(len(method_candidates)))
        if method.is_possible(schedule, env.axis_pool):
            method_instance = method.get_random(
                env.axis_pool, env.schedule[env.computed_arg]
            )
            schedule.append(method_instance)
            method_candidates = self.api_description.copy()
        return schedule, method_candidates

    def choose_random(self, _=None) -> tvm.module.Module:
        while True:
            env = self.create_schedule()
            schedule: list[Method] = []
            method_candidates = self.api_description.copy()
            while len(schedule) < self.min_length:
                schedule, method_candidates = self._add_next_method(
                    method_candidates, schedule, env
                )
                if not method_candidates:
                    break
            if len(schedule) < self.min_length:
                self.failed_random_generations += 1
                continue
            desired_length = random.randint(self.min_length, self.max_length)
            method_candidates = self.api_description.copy()
            while len(schedule) < desired_length:
                schedule, method_candidates = self._add_next_method(
                    method_candidates, schedule, env
                )
                if not method_candidates:
                    break
            break
        self.successful_random_generations += 1
        # print("Try schedule:", schedule)
        return self.finish_schedule(env)
