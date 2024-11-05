import json
import os

import matplotlib.pyplot as plt


def plot_results():
    with open(os.path.join("results", "results.json"), "r") as f:
        results = json.loads(f.read())

    for benchmark_name, scores in results.items():
        if benchmark_name == "total_time_schedgehammer":
            continue
        plt.figure()
        plt.plot(scores, label=benchmark_name)
        plt.xscale("log")
        plt.xlabel("function evaluations")
        plt.ylabel("cost")
        plt.title(benchmark_name)
        plt.savefig(f"results/{benchmark_name}.png")
    # plt.plot(results["total_time_schedgehammer"], label="total_time")
    # plt.legend()
    # plt.show()


if __name__ == "__main__":
    plot_results()
