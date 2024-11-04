import json
import os

import matplotlib.pyplot as plt


def plot_results():
    with open(os.path.join("results", "results.json"), "r") as f:
        results = json.loads(f.read())

    for benchmark_name, scores in results.items():
        if benchmark_name == "total_time":
            continue
        plt.plot(scores, label=benchmark_name)
    plt.plot(results["total_time"], label="total_time")
    plt.xscale("log")
    plt.yscale("log")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    plot_results()
