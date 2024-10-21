from collections import defaultdict

import matplotlib.pyplot as plt
from tune import load_params, tune

if __name__ == "__main__":
    params_ungrouped = load_params("params.yaml")
    params_grouped = load_params("params_grouped.yaml")
    budgets = []
    for exp in range(-5, -8, -1):
        for i in range(1, 10):
            budgets.append((i, exp))
    grouped = []
    ungrouped = []
    for budget in budgets:
        grouped.append(tune(params_grouped, budget[0], budget[1]))
        ungrouped.append(tune(params_ungrouped, budget[0], budget[1]))
    print(grouped)
    print(budgets)
    budgets = [b[0] * 10 ** b[1] for b in budgets]

    plt.plot(budgets, grouped, "bo", label="Grouped")
    plt.plot(budgets, ungrouped, "ro", label="Ungrouped")
    plt.show()
