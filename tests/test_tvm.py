import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy
import numpy as np
import tvm
from matplotlib import pyplot as plt
from tvm import auto_scheduler, te
from tvm.auto_scheduler.measure import PythonBasedMeasureCallback

from schedgehammer.problem import Problem
from schedgehammer.random_search import RandomSearch
from schedgehammer.schedule_type import ScheduleEnvironment, ScheduleParam
from schedgehammer.tuner import EvalBudget

M = 512
K = 512
N = 512

DTYPE = "float32"
ITERATIONS = 60  # If >63 limit ansors iterations else it will crash
RUNS = 3

k = te.reduce_axis((0, K), "k")
A = te.placeholder((M, K), name="A")
B = te.placeholder((K, N), name="B")
C = te.compute((M, N), lambda m, n: te.sum(A[m, k] * B[k, n], axis=k), name="C")

# Default schedule
s = te.create_schedule(C.op)
res = s[C].reorder(C.op.axis[0], C.op.axis[1], C.op.reduce_axis[0])
print(res)
