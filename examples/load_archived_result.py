import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
##############################################################

from schedgehammer.benchmark import ArchivedResult

BENCHMARKS = [
    "spmm",
    "spmv",
    "sddmm",
    "mttkrp",
    "ttv",
    "asum",
    "harris",
    "kmeans",
    "stencil",
]

for benchmark in BENCHMARKS:
    archived_res = ArchivedResult()
    archived_res.load_runs(f"results/ab3cdd2/{benchmark}/runs")
    archived_res.load_runs(f"results/opentuner/{benchmark}")
    archived_res.plot(f"results/opentuner/{benchmark}.png")
