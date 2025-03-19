import numpy
import numpy as np
import tvm
from matplotlib import pyplot as plt
from tvm import te

from tvm_api import REORDER, SPLIT, TILE
from schedgehammer.schedules.schedule_type import ScheduleParam, ScheduleContext

M = 512
K = 512
N = 512

DTYPE = "float32"

costs_per_schedule = []


def plot_results_from_several_runs(results, label) -> range:
    """Plot results from multiple runs and returns x values"""
    zipped_results = list(zip(*results))
    means = [np.mean(x) for x in zipped_results]
    mins = [np.min(x) for x in zipped_results]
    maxs = [np.max(x) for x in zipped_results]
    xs = range(len(zipped_results))
    plt.plot(xs, means, label=label)
    plt.fill_between(xs, mins, maxs, alpha=0.3)
    return xs


def create_schedule() -> ScheduleContext:
    k = te.reduce_axis((0, K), "k")
    A = te.placeholder((M, K), name="A")
    B = te.placeholder((K, N), name="B")
    C = te.compute((M, N), lambda m, n: te.sum(A[m, k] * B[k, n], axis=k), name="C")

    # Default schedule
    s = te.create_schedule(C.op)

    return ScheduleContext(
        [C.op.axis[0], C.op.axis[1], C.op.reduce_axis[0]],
        {
            'schedule': s,
            'tensor': C,
            'alltensors': [A, B, C],
        }
    )


def cost_function(config):
    dev = tvm.device("llvm", 0)

    # Random generated tensor for testing
    a = tvm.nd.array(numpy.random.rand(M, K).astype(DTYPE), dev)
    b = tvm.nd.array(numpy.random.rand(K, N).astype(DTYPE), dev)
    c = tvm.nd.array(numpy.zeros((M, N), dtype=DTYPE), dev)

    func: tvm.module.Module = config["schedule"]
    evaluator = func.time_evaluator(func.entry_name, dev, repeat=1)

    result = evaluator(a, b, c).mean
    correct_answer = numpy.dot(a.asnumpy(), b.asnumpy())
    c_numpyfied = c.asnumpy()
    assert np.allclose(
        c_numpyfied, correct_answer
    )  # test if same shape, elements have close enough values

    costs_per_schedule[-1].append(result)
    return result


def finish_schedule(ctx: ScheduleContext):
    return tvm.build(
        ctx.environment['schedule'],
        ctx.environment['alltensors'],
        name="anything",
    )


if __name__ == "__main__":
    param = ScheduleParam(
        create_schedule,
        finish_schedule,
        2,
        10,
        api_description=[TILE, SPLIT, REORDER],
    )
    best_cost = float("inf")
    for schedule_num in range(200):
        schedule = param.choose_random()
        tree = param.last_generated_tree
        costs_per_schedule.append([len(tree.operations)])
        for variant_num in range(1):
            print(variant_num, "/", schedule_num)
            tree.randomly_tweak_primitive_params()
            cost = cost_function({"schedule": param.translate_for_evaluation(tree)})
            if cost < best_cost:
                best_cost = cost
            print(best_cost, cost)
    plt.figure()
    # Group costs by schedule length and calculate averages of best 25%
    costs_by_length = {}
    for costs in costs_per_schedule:
        length = costs[0]
        if length not in costs_by_length:
            costs_by_length[length] = []
        costs_by_length[length].extend(costs[1:])

    lengths = sorted(costs_by_length.keys())
    avg_costs = []
    for l in lengths:
        costs = sorted(costs_by_length[l])
        num_best = max(1, len(costs) // 4)  # Take top 25%
        best_costs = costs[:num_best]
        avg_costs.append(sum(best_costs) / len(best_costs))

    plt.plot(lengths, avg_costs, "o-")
    plt.xlabel("Schedule Length")
    plt.ylabel("Average Cost (Best 25%)")
    plt.yscale("log")
    plt.show()
