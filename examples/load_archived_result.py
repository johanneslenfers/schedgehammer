import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
##############################################################

from schedgehammer.benchmark import ArchivedResult

archived_res = ArchivedResult()
archived_res.load_runs("results/asum/runs")
archived_res.load_runs("results/_constrained3/asum/runs")
archived_res.plot("test.png")
