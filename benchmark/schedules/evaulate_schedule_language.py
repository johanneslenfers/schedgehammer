from typing import Any, Callable

import numpy as np
from matplotlib import pyplot as plt

from schedgehammer.problem import Problem
from schedgehammer.schedules.schedule_genetic_tuner import ScheduleGeneticTuner
from schedgehammer.schedules.schedule_random_search import ScheduleRandomSearch
from schedgehammer.schedules.schedule_type import (
    Operation,
    ScheduleContext,
    ScheduleParam,
)
from schedgehammer.tuner import EvalBudget, ParameterConfiguration

TUNERS = {
    "Genetic Tuner": ScheduleGeneticTuner,
    "Random Search": ScheduleRandomSearch,
}


def plot_results_from_several_runs(results, label) -> range:
    """Plot results from multiple runs and returns x values"""
    zipped_results = list(zip(*results))
    means = [np.mean(x) for x in zipped_results]
    lower_percentile = [np.percentile(x, 25) for x in zipped_results]
    upper_percentile = [np.percentile(x, 75) for x in zipped_results]
    xs = range(len(zipped_results))
    plt.plot(xs, means, label=label)
    plt.fill_between(
        xs,
        lower_percentile,
        upper_percentile,
        alpha=0.3,
    )
    return xs


def evaluate_problem_for_schedule_language(
    problem_name: str,
    language_name: str,
    create_schedule_function: Callable[[], ScheduleContext],
    cost_function: Callable[[ParameterConfiguration], float],
    finish_schedule_function: Callable[[ScheduleContext], Any],
    api_description: list[Operation],
    first_operation_blacklist: list[Operation] = [],
    rival_tuners: dict[str, Callable[[int, int], list[list[float]]]] = {},
    other_results: dict[str, float] = {},
    runs: int = 20,
    iterations: int = 1000,
    schedule_config_key: str = "schedule",
) -> None:
    base_schedule_ctx = create_schedule_function()
    config = finish_schedule_function(base_schedule_ctx)
    baseline_cost = cost_function({schedule_config_key: config})
    print(
        f"\033[94mBaseline cost for {language_name} on {problem_name}: {baseline_cost}\033[0m"
    )
    rival_results = {}
    results_by_tuner = {}
    for name, func in rival_tuners.items():
        results = func(iterations, runs)
        rival_results[name] = results
    for run in range(runs):
        print(f"\033[95mRUN: {run}\033[0m")
        for tuner_name, tuner_class in TUNERS.items():
            for min_, max_ in [(1, 50), (1, 10)]:
                key = f"{tuner_name} min={min_} max={max_}"
                tuner = tuner_class()
                param = ScheduleParam(
                    create_schedule_function,
                    finish_schedule_function,
                    min_,
                    max_,
                    api_description=api_description,
                    first_operation_blacklist=first_operation_blacklist,
                )
                result = tuner.tune(
                    problem=Problem(
                        problem_name,
                        {schedule_config_key: param},
                        cost_function,
                        [],
                        init_solver=False,
                    ),
                    budgets=[EvalBudget(iterations)],
                )
                if key not in results_by_tuner:
                    results_by_tuner[key] = []
                results_by_tuner[key].append(result.best_score_list())
            print(
                f"\033[94m{tuner_name} best score for {language_name} on {problem_name}: {min(result.best_score_list())}\033[0m"
            )
    plt.figure()
    for tuner_name, tuner_results in results_by_tuner.items():
        xs = plot_results_from_several_runs(tuner_results, tuner_name)
    plt.plot(xs, [baseline_cost] * len(xs), label="Baseline")
    for other_name, other_result in other_results.items():
        plt.plot(xs, [other_result] * len(xs), label=other_name)
    for rival_name, rival_results in rival_results.items():
        plot_results_from_several_runs(rival_results, rival_name)
    plt.xlabel("Cost function evaluations")
    plt.ylabel("Cost (s)")
    plt.title(f"{language_name} on {problem_name}")
    plt.yscale("log")
    plt.gca().yaxis.set_major_formatter(plt.ScalarFormatter())
    plt.gca().yaxis.get_major_formatter().set_scientific(False)
    plt.legend()

    import os

    results_dir = f"results/{language_name}"
    os.makedirs(results_dir, exist_ok=True)

    # Save the plot to the language-specific directory
    plt.savefig(f"{results_dir}/{problem_name}.png")
    plt.close()
