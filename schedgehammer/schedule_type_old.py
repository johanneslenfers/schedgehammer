from __future__ import annotations

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, ClassVar, Type

import tvm
from tvm.te import Tensor
from tvm.tir.expr import IterVar as TvmAxis

from schedgehammer.param_types import ExpIntParam, Param, ParamValue


@dataclass
class TvmAxisParam:
    consuming: bool  # Wether or not the axis gets removed from the pool
    force_inner: bool = False  # Wether or not it has to be the innermost axis


@dataclass
class TvmAxisPoolPermutationParam:
    pass


MethodParamType = TvmAxisParam | TvmAxisPoolPermutationParam | Param
MethodParam = TvmAxis | list[TvmAxis] | ParamValue


def _pretty_print_schedule(schedule: list[Method]) -> None:
    print("\n\n### Schedule ###")
    for method in schedule:
        print("->", method.name)
        for param_name, param_value in method.params.items():
            print(f"  {param_name}: {param_value}")

    print("### End of Schedule ###")


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
    # Attributes of concrete method application
    params: dict[str, MethodParam]

    @classmethod
    @abstractmethod
    def is_possible(
        cls, previous_steps: list[Method], current_axis_pool: list[TvmAxis]
    ) -> bool:
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
                if param_type.force_inner:
                    axis_index = len(current_axis_pool) - 1
                else:
                    axis_index = random.randint(0, len(current_axis_pool) - 1)
                if param_type.consuming:
                    param_val = current_axis_pool.pop(axis_index)
                else:
                    param_val = current_axis_pool[axis_index]
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
        new_axes = method.apply(tensor)
        current_axis_pool.extend(new_axes)
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
class Vectorize(Method):
    name = "vectorize"
    param_types = {
        "axis": TvmAxisParam(consuming=True, force_inner=True),
    }

    @classmethod
    def is_possible(
        cls, previous_steps: list[Method], current_axis_pool: list[TvmAxis]
    ) -> bool:
        """
        Assumptions:
            - Vectorization requires at least one axis to be present
            - Most inner axis can't be reduce axis
            - Previous step has to be reordering so that we can be sure we get to
              vectorize the most inner axis
        """
        num_axes_available = len(current_axis_pool)
        if num_axes_available < 1:
            return False
        if current_axis_pool[-1].iter_type == 2:  # if reduce axis
            return False
        if previous_steps and previous_steps[-1].name != Reorder.name:
            return False
        return True

    def apply(self, tensor):
        tensor.vectorize(self.params["axis"])
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
    api_description: list[Type[Method]] = field(
        default_factory=lambda: [Tile, Split, Reorder, Vectorize]
    )
    terminating_methods: list[Type[Method]] = field(default_factory=lambda: [Vectorize])
    genetic_tuning_mode: bool = False
    current_population: list[tuple[list[Method], float]] = field(default_factory=list)
    population_size: int = 6
    elitism_share: float = 0.1
    reproduction_share: float = 0.3
    crossover_prob: float = 0.5
    mutation_prob: float = 0.1
    local_mutation: bool = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.genetic_tuning_mode:
            for _ in range(self.population_size):
                env, schedule = self.create_random_schedule()
                compiled_schedule = self.finish_schedule(env)
                cost = self.cost_function({"schedule": compiled_schedule})
                self.current_population.append((schedule, cost))

    def try_appending_method(
        self,
        schedule: list[Method],
        method_candidates: list[Type[Method]],
        env: ScheduleEnvironment,
    ) -> tuple[list[Method], list[Type[Method]], Type[Method] | None]:
        """
        Tries to append one of the method candidates to the schedule.
        Returns:
            - Updated schedule
            - Updated method candidates
            - Chosen method type
        """
        method = method_candidates.pop(random.randrange(len(method_candidates)))
        if method.is_possible(schedule, env.axis_pool):
            method_instance = method.get_random(
                env.axis_pool, env.schedule[env.computed_arg]
            )
            schedule.append(method_instance)
            if len(schedule) < self.min_length:
                method_candidates = [
                    method
                    for method in self.api_description
                    if method not in self.terminating_methods
                ]
            else:
                method_candidates = self.api_description.copy()
            return schedule, method_candidates, method
        return schedule, method_candidates, None

    def create_random_schedule(self) -> tuple[ScheduleEnvironment, list[Method]]:
        while True:
            env = self.create_schedule()
            schedule: list[Method] = []
            desired_length = random.randint(self.min_length, self.max_length)
            method_candidates = [
                method
                for method in self.api_description
                if method not in self.terminating_methods
            ]
            while len(schedule) < desired_length:
                schedule, method_candidates, method = self.try_appending_method(
                    schedule, method_candidates, env
                )
                if method in self.terminating_methods:
                    break
                if not method_candidates:
                    break
            if len(schedule) >= self.min_length:
                break
        return env, schedule

    def choose_random(self, _=None) -> tvm.module.Module:
        env, schedule = self.create_random_schedule()
        _pretty_print_schedule(schedule)
        return self.finish_schedule(env)
