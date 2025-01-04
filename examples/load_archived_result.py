import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
##############################################################

from schedgehammer.benchmark import ArchivedResult

BENCHMARKS = [
    "mttkrp"
]

for benchmark in BENCHMARKS:
    archived_res = ArchivedResult()
    archived_res.load_runs(f"results/13dfef0/{benchmark}/constrained/runs", [
        'GeneticTuner with constraints', 'RandomSearch with constraints',
    ])
    archived_res.load_runs(f"results/13dfef0/{benchmark}/unconstrained/runs", [
        'GeneticTuner without constraints', 'RandomSearch without constraints',
    ])
    archived_res.load_runs(f"results/opentuner/{benchmark}")
    archived_res.load_runs(f"results/atf/{benchmark}/csv")

    archived_res.rename('ATF', 'pyATF')

    archived_res.plot(f"results/atf/{benchmark}/{benchmark}.png", benchmark)
