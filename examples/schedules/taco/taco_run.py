import json
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../../.."))

from examples.schedules.taco.gemm import TacoProblem
from examples.schedules.taco.mttkrp import TacoMttkrpProblem
from examples.schedules.taco.spmv import TacoSpmvProblem
from schedgehammer.benchmark import benchmark
from schedgehammer.schedules.schedule_genetic_tuner import ScheduleGeneticTuner
from schedgehammer.schedules.schedule_random_search import ScheduleRandomSearch
from schedgehammer.tuner import EvalBudget

def main():
    assert len(sys.argv) >= 2
    problem_clss = {
        "gemm": TacoProblem,
        "spmv": TacoSpmvProblem,
        "mttkrp": TacoMttkrpProblem,
    }

    assert sys.argv[1] in problem_clss
    problem_cls = problem_clss[sys.argv[1]]
    problem = problem_cls()

    mode = sys.argv[2] if len(sys.argv) > 2 else None

    if mode == "baseline":
        s = problem.create_schedule()
        val = problem.finish_schedule(s)
        time = problem.cost_function({"schedule": val})
        results_dir = os.environ.get("RESULTS_DIR", "results")
        os.makedirs(f"{results_dir}/base", exist_ok=True)
        with open(f"{results_dir}/base/{problem.name}.json", "w") as f:
            json.dump(time, f)
    else:
        budget = int(os.environ.get("BUDGET", 100))
        runs = int(os.environ.get("RUNS", 1))
        results_dir = os.environ.get("RESULTS_DIR", "results")
        benchmark(
            problem_cls,
            [EvalBudget(budget)],
            {
                "genetic_tuner": ScheduleGeneticTuner(),
                "random_tuner": ScheduleRandomSearch(),
            },
            f"{results_dir}/taco/{problem.name}",
            runs,  # Number of repetitions for each tuner
            True,
            1,  # Run sequentially (parallel=1) to allow proper interruption
        )


if __name__ == "__main__":
    main()

