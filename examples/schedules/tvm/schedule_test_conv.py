import numpy as np
import tvm
import tvm.topi as topi
from matplotlib import pyplot as plt
from tvm import auto_scheduler, te
from tvm.auto_scheduler.measure import PythonBasedMeasureCallback
from tvm_api import REORDER, SPLIT, TILE

from schedgehammer.problem import Problem
from schedgehammer.schedules.schedule_genetic_tuner import ScheduleGeneticTuner
from schedgehammer.schedules.schedule_random_search import ScheduleRandomSearch
from schedgehammer.schedules.schedule_type import ScheduleContext, ScheduleParam
from schedgehammer.tuner import EvalBudget

N, H, W, C_in = 1, 64, 64, 3  # Batch size, height, width, input channels
K_h, K_w, C_out = 3, 3, 128  # Kernel height, kernel width, output channels
stride, padding = 1, 1


DTYPE = "float32"
ITERATIONS = 100  # If >63 limit ansors iterations else it will crash
RUNS = 9

results_genetic = []
results_random = []
ansor_results = []


def plot_results_from_several_runs(results, label) -> range:
    """Plot results from multiple runs and returns x values"""
    zipped_results = list(zip(*results))
    means = [np.mean(x) for x in zipped_results]
    lower_percentile = [np.percentile(x, 5) for x in zipped_results]
    upper_percentile = [np.percentile(x, 95) for x in zipped_results]
    xs = range(len(zipped_results))
    plt.plot(xs, means, label=label)
    plt.fill_between(
        xs,
        lower_percentile,
        upper_percentile,
        alpha=0.3,
    )
    return xs


class StoreResultCallback(PythonBasedMeasureCallback):
    def callback(self, policy, inputs, results):
        for result in results[0:]:
            cost = float(result.costs[0])
            if not ansor_results[-1] or cost < ansor_results[-1][-1]:
                ansor_results[-1].append(cost)
            else:
                ansor_results[-1].append(ansor_results[-1][-1])


def create_schedule() -> ScheduleContext:
    Input = te.placeholder((N, H, W, C_in), dtype=DTYPE, name="Input")
    Kernel = te.placeholder((K_h, K_w, C_in, C_out), dtype=DTYPE, name="Kernel")

    # Use TOPI to define the convolution operation
    Output = topi.nn.conv2d_nhwc(
        Input, Kernel, stride=stride, padding=padding, dilation=1, out_dtype=DTYPE
    )
    s = te.create_schedule(Output.op)

    # # Apply optimizations
    # block_x = te.thread_axis("blockIdx.x")
    # thread_x = te.thread_axis("threadIdx.x")

    # # Split and reorder loops for better memory access
    # n, h, w, c = s[Output].op.axis
    # ho, hi = s[Output].split(h, factor=8)
    # wo, wi = s[Output].split(w, factor=8)
    # co, ci = s[Output].split(c, factor=16)

    # s[Output].reorder(n, ho, wo, co, hi, wi, ci)

    # # Parallelize computation
    # s[Output].bind(ho, block_x)
    # s[Output].bind(wi, thread_x)

    return ScheduleContext(
        [
            Output.op.axis[0],
            Output.op.axis[1],
            Output.op.axis[2],
            Output.op.axis[3],
            Output.op.reduce_axis[0],
            Output.op.reduce_axis[1],
            Output.op.reduce_axis[2],
        ],
        {
            "schedule": s,
            "tensor": Output,
            "alltensors": [Input, Kernel, Output],
        },
    )


def cost_function(config):
    input_data = np.random.randn(N, H, W, C_in).astype(DTYPE)
    kernel_data = np.random.randn(K_h, K_w, C_in, C_out).astype(DTYPE)

    # Allocate TVM NDArray
    dev = tvm.device("llvm", 0)
    input_tvm = tvm.nd.array(input_data, dev)
    kernel_tvm = tvm.nd.array(kernel_data, dev)
    output_tvm = tvm.nd.array(np.zeros((N, H, W, C_out), dtype=DTYPE), dev)

    # Run the optimized convolution
    func: tvm.module.Module = config["schedule"]
    evaluator = func.time_evaluator(func.entry_name, dev, repeat=1)
    result = evaluator(input_tvm, kernel_tvm, output_tvm).mean

    print("COST:", result)
    return result


def finish_schedule(ctx: ScheduleContext):
    return tvm.build(
        ctx.environment["schedule"],
        ctx.environment["alltensors"],
        target="llvm",
        name="anything",
    )


def get_ansor_results():
    @auto_scheduler.register_workload
    def create_task_func():
        Input = te.placeholder((N, H, W, C_in), dtype=DTYPE, name="Input")
        Kernel = te.placeholder((K_h, K_w, C_in, C_out), dtype=DTYPE, name="Kernel")

        # Use TOPI to define the convolution operation
        Output = topi.nn.conv2d_nhwc(
            Input,
            Kernel,
            stride=stride,
            padding=padding,
            dilation=1,
            out_dtype=DTYPE,
        )
        return [Input, Kernel, Output]

    # Create the search task
    target = tvm.target.Target("llvm")
    for _ in range(RUNS):
        ansor_results.append([])
        task = auto_scheduler.SearchTask(func=create_task_func, target=target)

        tuning_options = auto_scheduler.TuningOptions(
            num_measure_trials=min(ITERATIONS, 63),
            measure_callbacks=[
                auto_scheduler.RecordToFile("conv2d.json"),
                StoreResultCallback(),
            ],
            verbose=2,
        )

        # Begin tuning process
        task.tune(tuning_options)

        # Apply the best schedule and build the function
        sch, args = task.apply_best("conv2d.json")
        func = tvm.build(sch, args, target)

        dev = tvm.device("llvm", 0)
        # Create sample input arrays
        input_data = np.random.uniform(size=(N, H, W, C_in)).astype(DTYPE)
        kernel_data = np.random.uniform(size=(K_h, K_w, C_in, C_out)).astype(DTYPE)
        output_data = np.zeros((N, H, W, C_out), dtype=DTYPE)

        # Create TVM NDArray
        input_tvm = tvm.nd.array(input_data, dev)
        kernel_tvm = tvm.nd.array(kernel_data, dev)
        output_tvm = tvm.nd.array(output_data, dev)

        evaluator = func.time_evaluator(func.entry_name, dev, number=5)
        exec_time = evaluator(input_tvm, kernel_tvm, output_tvm).mean
        print("Ansor execution time: %.3f ms" % (exec_time * 1e3))
        ansor_results[-1][-1] = (
            exec_time  # Make sure the real execution time is not higher than calculated by ansor
        )


# def get_baseline() -> float:
#     # Find time of unchanged schedule
#     env = create_schedule()
#     func = tvm.build(
#         env.schedule, env.static_args + [env.computed_arg], name="anything"
#     )
#     dev = tvm.device("llvm", 0)
#     input_data = np.random.randn(N, H, W, C_in).astype(DTYPE)
#     kernel_data = np.random.randn(K_h, K_w, C_in, C_out).astype(DTYPE)
#     Output = topi.nn.conv2d_nhwc(
#         Input, Kernel, stride=stride, padding=padding, dilation=1, out_dtype=DTYPE
#     )
#     evaluator = func.time_evaluator(func.entry_name, dev, repeat=3)
#     result = evaluator(a, b, c).mean
#     return result


if __name__ == "__main__":
    # get_ansor_results()
    for result_list, tuner_class in [
        (results_genetic, ScheduleGeneticTuner),
        (results_random, ScheduleRandomSearch),
    ]:
        tuner = tuner_class()
        for run in range(RUNS):
            print("\033[95mRUN:", run, "\033[0m")
            param = ScheduleParam(
                create_schedule,
                finish_schedule,
                2,
                13,
                api_description=[TILE, SPLIT, REORDER],
            )
            result = tuner.tune(
                problem=Problem(
                    "schedge",
                    {"schedule": param},
                    cost_function,
                    [],
                    init_solver=False,
                ),
                budgets=[EvalBudget(ITERATIONS)],
            )
            result_list.append(result.best_score_list())
    # baseline_score = get_baseline()
    # print("Baseline:", baseline_score)
    plt.figure()
    plot_results_from_several_runs(results_genetic, "Genetic Search")
    plot_results_from_several_runs(results_random, "Random Search")
    xs = plot_results_from_several_runs(ansor_results, "Ansor")
    # plt.plot(xs, [baseline_score] * len(xs), label="Baseline")
    plt.xlabel("function evaluations")
    plt.ylabel("cost")
    plt.yscale("log")
    plt.gca().yaxis.set_major_formatter(plt.ScalarFormatter())
    plt.gca().yaxis.get_major_formatter().set_scientific(False)
    plt.legend()
    plt.show()
