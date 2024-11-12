import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
##############################################################
from pathlib import Path

from schedgehammer.benchmark import ArchivedResult

archived_res = ArchivedResult(Path("results/asum/runs"))
archived_res.plot()
