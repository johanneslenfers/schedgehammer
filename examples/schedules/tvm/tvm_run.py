import json
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../../.."))

from examples.schedules.tvm.conv_2d import Conv2DProblem
from examples.schedules.tvm.mttkrp import MttkrpProblem
from examples.schedules.tvm.mm import MMProblem
from examples.schedules.tvm.tvm_schedule_problem import TVMScheduleProblem
from schedgehammer.schedules.schedule_genetic_tuner import ScheduleGeneticTuner
from schedgehammer.schedules.schedule_random_search import ScheduleRandomSearch
from schedgehammer.benchmark import benchmark
from schedgehammer.tuner import EvalBudget


def main():
    assert len(sys.argv) >= 2
    problem_clss = {
        "mm": MMProblem,
        "conv2d": Conv2DProblem,
        "mttkrp": MttkrpProblem,
    }

    assert sys.argv[1] in problem_clss
    problem_cls = problem_clss[sys.argv[1]]
    problem: TVMScheduleProblem = problem_cls()

    mode = sys.argv[2] if len(sys.argv) > 2 else None

    if mode == 'ansor':
        results = problem.get_ansor_results(63, 1)
        with open(f'results/ansor/{problem.name}.json', 'w') as f:
            json.dump(results, f)
    elif mode == 'baseline':
        s = problem.create_schedule()
        val = problem.finish_schedule(s)
        time = problem.cost_function({"schedule": val})
        with open(f'results/base/{problem.name}.json', 'w') as f:
            json.dump(time, f)
    else:
        benchmark(
            problem_cls,
            [EvalBudget(100)],
            {
                "genetic_tuner": ScheduleGeneticTuner(),
                "random_tuner": ScheduleRandomSearch(),
            },
            f"results/tvm/{problem.name}",
            15,
            True,
            16,
        )


if __name__ == "__main__":
    main()
