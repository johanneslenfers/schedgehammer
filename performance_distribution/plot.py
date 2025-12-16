import json
import os
import sys
from functools import cmp_to_key

import matplotlib.pyplot as plt
import numpy as np

PATH = sys.argv[1]
with open(PATH, "r") as f:
    results = json.load(f)

# Determine the fastest runtime for each program
program_metrics = []
for i, program_runtimes in enumerate(results):
    if not program_runtimes:  # Skip empty lists
        print("skipping empty list")
        continue

    fastest_runtime = min(program_runtimes)
    worst_runtime = max(program_runtimes)
    program_metrics.append(
        {
            "index": i,
            "runtimes": program_runtimes,
            "fastest": fastest_runtime,
            "worst": worst_runtime,
        }
    )

# Sort programs by fastest runtime, with tolerance check for similar performances
# If within tolerance, compare by worst runtime
TOLERANCE = 0.05  # 5% tolerance for "similar" runtimes


def compare_two_elements(a, b):
    return -1 if a["fastest"] < b["fastest"] else 1
    # """
    # Compare two elements directly.
    # Return: -1 if a should come before b
    #         0 if a and b are equal
    #         1 if b should come before a
    # """
    # if a["fastest"] < b["fastest"]:
    #     if a["fastest"] * (1 + TOLERANCE) < b["fastest"]:
    #         return -1
    #     else:
    #         if a["worst"] > b["worst"]:
    #             return 1
    #         else:
    #             return -1
    # else:
    #     if b["fastest"] * (1 + TOLERANCE) < a["fastest"]:
    #             return 1
    #         else:
    #             if a["worst"] > b["worst"]:
    #                 return 1
    #             else:
    #                 return -1


sorted_programs = sorted(program_metrics, key=cmp_to_key(compare_two_elements))

plt.style.use("default")
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.facecolor"] = "white"
plt.rcParams["figure.dpi"] = 300
plt.rcParams["font.size"] = 12
plt.rcParams["axes.labelsize"] = 14
plt.rcParams["axes.titlesize"] = 16
plt.rcParams["xtick.labelsize"] = 22
plt.rcParams["ytick.labelsize"] = 22
plt.rcParams["legend.fontsize"] = 12

plt.figure(figsize=(15, 8))

# Extract data for plotting
y_positions = range(len(sorted_programs))
for i, program in enumerate(sorted_programs):
    runtimes = program["runtimes"]
    # Create horizontal dots for each runtime
    x_positions = [i] * len(runtimes)
    # Color based on runtime value (y-position), green for low values to red for high values
    plt.scatter(x_positions, runtimes, alpha=0.6, s=1, c=runtimes, cmap="RdYlGn_r")

plt.ylabel("Runtime (seconds)", fontsize=22)
plt.xlabel("Schedule Index (sorted by performance)", fontsize=22)
plt.title(
    "Runtime Distribution over primitive parameter mutations for different schedules",
    fontsize=22,
)
plt.grid(True, alpha=0.3, color="gray", linestyle="-", linewidth=0.5)
plt.yscale("log")

# Add some statistics
fastest_overall = min(program["fastest"] for program in sorted_programs)
plt.axhline(
    y=fastest_overall,
    color="green",
    linestyle="--",
    alpha=0.7,
    linewidth=2,
    label=f"Fastest overall: {fastest_overall:.6f}s",
)


plt.legend(
    framealpha=0.9,
    fancybox=True,
    shadow=True,
    fontsize=22,
    loc="lower right",
    bbox_to_anchor=(1.0, 0.03),
)
plt.tight_layout()
if len(sys.argv) > 2:
    # Save to file if output path provided
    output_path = sys.argv[2]
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Plot saved to {output_path}")
else:
    # Show interactively if no output path
    plt.show()
