# Only needed since this is in the same repo as schedgehammer.
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
##############################################################
import tvm
from conv_2d import (
    conv_2d_cost_function,
    create_2d_conv_schedule,
    get_ansor_conv_2d_results,
)
from dense_layer import (
    create_dense_layer_schedule,
    dense_layer_cost_function,
    get_ansor_dense_layer_results,
)
from evaulate_schedule_language import evaluate_problem_for_schedule_language
from matplotlib import pyplot as plt
from mm import (
    create_mm_schedule,
    get_ansor_mm_results,
    get_best_mm_blocking_baseline,
    mm_cost_function,
)
from mttkrp import (
    create_mttkrp_schedule,
    get_ansor_mttkrp_results,
    mttkrp_cost_function,
)
from tvm import auto_scheduler, te
from tvm.auto_scheduler.measure import PythonBasedMeasureCallback
from tvm_api import REORDER, SPLIT, TILE

from schedgehammer.problem import Problem
from schedgehammer.schedules.schedule_genetic_tuner import ScheduleGeneticTuner
from schedgehammer.schedules.schedule_random_search import ScheduleRandomSearch
from schedgehammer.schedules.schedule_type import ScheduleContext, ScheduleParam
from schedgehammer.tuner import EvalBudget, Tuner

DEFAULT_RUNS = 7
DEFAULT_ITERATIONS = 200


def finish_schedule(ctx: ScheduleContext):
    return tvm.build(
        ctx.environment["schedule"],
        ctx.environment["alltensors"],
        name="anything",
    )


def evaluate_mm_schedule(runs=DEFAULT_RUNS, iterations=DEFAULT_ITERATIONS):
    evaluate_problem_for_schedule_language(
        "Matrix Multiplication",
        "tvm",
        create_mm_schedule,
        mm_cost_function,
        finish_schedule,
        [TILE, SPLIT, REORDER],
        rival_tuners={
            "Ansor": get_ansor_mm_results,
        },
        other_results={
            "Optimized Block Schedule": get_best_mm_blocking_baseline(),
        },
        runs=runs,
        iterations=iterations,
    )


def evaluate_mttkrp_schedule(runs=DEFAULT_RUNS, iterations=DEFAULT_ITERATIONS):
    evaluate_problem_for_schedule_language(
        "Matricized tensor times Khatri-Rao product",
        "tvm",
        create_mttkrp_schedule,
        mttkrp_cost_function,
        finish_schedule,
        [TILE, SPLIT, REORDER],
        runs=runs,
        iterations=iterations,
        rival_tuners={
            "Ansor": get_ansor_mttkrp_results,
        },
    )


def evaluate_conv_2d_schedule(runs=DEFAULT_RUNS, iterations=DEFAULT_ITERATIONS):
    evaluate_problem_for_schedule_language(
        "Convolution 2D",
        "tvm",
        create_2d_conv_schedule,
        conv_2d_cost_function,
        finish_schedule,
        [TILE, SPLIT, REORDER],
        runs=runs,
        iterations=iterations,
        rival_tuners={
            "Ansor": get_ansor_conv_2d_results,
        },
    )


def evaluate_dense_layer_schedule(runs=DEFAULT_RUNS, iterations=DEFAULT_ITERATIONS):
    evaluate_problem_for_schedule_language(
        "Dense Layer",
        "tvm",
        create_dense_layer_schedule,
        dense_layer_cost_function,
        finish_schedule,
        [TILE, SPLIT, REORDER],
        runs=runs,
        iterations=iterations,
        rival_tuners={
            "Ansor": get_ansor_dense_layer_results,
        },
    )


def full_benchmark():
    evaluate_mm_schedule()
    evaluate_conv_2d_schedule()
    evaluate_mttkrp_schedule()


if __name__ == "__main__":
    full_benchmark()
