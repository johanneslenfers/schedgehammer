import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
##############################################################

from schedgehammer.benchmark import ArchivedResult

archived_res = ArchivedResult()
archived_res.load_runs("results/harris/runs")
archived_res.load_runs("results/kmeans/runs")
archived_res.load_runs("results/mttkrp/runs")
archived_res.load_runs("results/sddmm/runs")
archived_res.load_runs("results/spmm/runs")
archived_res.load_runs("results/spmv/runs")
archived_res.load_runs("results/stencil/runs")
archived_res.load_runs("results/ttv/runs")
archived_res.plot("test.png")
