#!/usr/bin/env python3
"""
Compatible with Python 3.10.16.

Scan CatBench benchmark results and plot *performance evolution* as the
best-so-far curve over evaluations (running minimum), then aggregate across runs
(median + 95% bootstrap CI across runs at each x).

IMPORTANT BEHAVIOR:
- If a CSV entry has a missing / non-numeric value, it is treated as +infinity
  (i.e., worst possible performance).
- Performance evolution is computed PER RUN first (running minimum),
  THEN median and CI are computed across runs.

Benchmarks: harris, mttkrp, spmv
Outputs:
  - performance_evolution_catbench.png
  - performance_evolution_catbench.pdf
"""

from __future__ import annotations

import argparse
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')

BENCHMARKS = ["harris", "mttkrp", "spmv"]
METHODS_ORDER = ["pyatf", "opentuner", "randomSearch", "geneticTuner"]

METHOD_COLORS = {
    "pyatf": "#1f77b4",
    "opentuner": "#ff7f0e",
    "randomSearch": "#2ca02c",
    "geneticTuner": "#d62728",
}

METHOD_LABELS = {
    "pyatf": "pyatf",
    "opentuner": "opentuner",
    "randomSearch": "randomSearch",
    "geneticTuner": "geneticTuner",
}

INTMAX = float(np.iinfo(np.int64).max)


@dataclass
class RunSeries:
    x: np.ndarray
    y: np.ndarray
    source: str


def warn(msg: str) -> None:
    warnings.warn(msg, RuntimeWarning, stacklevel=2)


def find_csvs_recursive(root: Path) -> List[Path]:
    if not root.exists() or not root.is_dir():
        return []
    return sorted(p for p in root.rglob("*.csv") if p.is_file())


def find_random_genetic_csvs(root: Path, benchmark: str) -> Dict[str, List[Path]]:
    runs_dir = root / benchmark / "runs"
    out = {"randomSearch": [], "geneticTuner": []}
    if not runs_dir.exists():
        return out

    for p in sorted(runs_dir.glob("*.csv")):
        name = p.name
        if re.match(r"(?i)^randomsearch.*\.csv$", name):
            out["randomSearch"].append(p)
        elif re.match(r"(?i)^genetictuner.*\.csv$", name):
            out["geneticTuner"].append(p)
    return out


def discover_files(root: Path, benchmark: str) -> Dict[str, List[Path]]:
    files = {m: [] for m in METHODS_ORDER}

    rg = find_random_genetic_csvs(root, benchmark)
    files["randomSearch"].extend(rg["randomSearch"])
    files["geneticTuner"].extend(rg["geneticTuner"])

    files["pyatf"].extend(find_csvs_recursive(root / f"pyatf_{benchmark}"))
    files["opentuner"].extend(find_csvs_recursive(root / f"opentuner_{benchmark}"))

    return files


def _sanitize_and_sort_xy(
    df: pd.DataFrame, xcol: str, ycol: str
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    - Missing x rows are dropped
    - Missing / non-numeric y values are replaced with +infinity
    """
    x = pd.to_numeric(df[xcol], errors="coerce")
    y = pd.to_numeric(df[ycol], errors="coerce").fillna(INTMAX)

    mask = x.notna()
    x = x[mask].astype(int)
    y = y[mask].astype(float)

    if len(x) == 0:
        return None

    order = np.argsort(x.to_numpy(), kind="mergesort")
    x_arr = x.to_numpy()[order]
    y_arr = y.to_numpy()[order]

    unique = len(np.unique(x_arr)) == len(x_arr)
    strictly_inc = len(x_arr) < 2 or np.all(np.diff(x_arr) > 0)

    if not (unique and strictly_inc):
        x_arr = np.arange(1, len(x_arr) + 1, dtype=int)

    return x_arr, y_arr


def running_min(y: np.ndarray) -> np.ndarray:
    """Best-so-far performance evolution."""
    return np.minimum.accumulate(y)


def load_run_series(method: str, csv_path: Path) -> Optional[RunSeries]:
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        warn(f"Skipping malformed CSV: {csv_path} ({e})")
        return None

    cols = {str(c).lower(): str(c) for c in df.columns}

    def pick(name: str) -> Optional[str]:
        return cols.get(name)

    if method in ("pyatf", "opentuner"):
        xcol, ycol = pick("evaluation"), pick("cost")
    else:
        xcol, ycol = pick("num_evaluation"), pick("score")

    if xcol is None or ycol is None:
        warn(f"Missing required columns in {csv_path}")
        return None

    xy = _sanitize_and_sort_xy(df, xcol, ycol)
    if xy is None:
        return None

    x_arr, y_raw = xy

    # Convert EACH RUN to performance evolution FIRST
    y_evo = running_min(y_raw)

    # Log-scale safety: keep strictly positive values
    pos = y_evo > 0
    if not np.any(pos):
        warn(f"All values non-positive after evolution: {csv_path}")
        return None

    return RunSeries(x=x_arr[pos], y=y_evo[pos], source=str(csv_path))


def collect_runs(method_files: Dict[str, List[Path]]) -> Dict[str, List[RunSeries]]:
    out: Dict[str, List[RunSeries]] = {}
    for method, files in method_files.items():
        runs = []
        for p in files:
            rs = load_run_series(method, p)
            if rs is not None:
                runs.append(rs)
        out[method] = runs
    return out


def align_runs_by_x(runs: List[RunSeries]) -> Tuple[np.ndarray, List[np.ndarray]]:
    xs = np.unique(np.concatenate([r.x for r in runs]).astype(int))
    xs.sort()

    ys_aligned = []
    for r in runs:
        y = np.full(xs.shape, np.nan)
        idx = np.searchsorted(xs, r.x)
        ok = (idx < len(xs)) & (xs[idx] == r.x)
        y[idx[ok]] = r.y[ok]
        ys_aligned.append(y)

    return xs, ys_aligned


def bootstrap_ci_median(values: np.ndarray, rng: np.random.Generator, n_resamples: int):
    idx = rng.integers(0, len(values), size=(n_resamples, len(values)))
    meds = np.median(values[idx], axis=1)
    return np.quantile(meds, [0.025, 0.975])


def compute_stats(xs, ys_aligned, n_resamples, seed):
    Y = np.vstack(ys_aligned)
    med = np.nanmedian(Y, axis=0)

    if Y.shape[0] < 2:
        return xs, med, None, None

    rng = np.random.default_rng(seed)
    lo = np.full_like(med, np.nan)
    hi = np.full_like(med, np.nan)

    for i in range(Y.shape[1]):
        col = Y[:, i]
        col = col[np.isfinite(col)]
        if len(col) >= 2:
            lo[i], hi[i] = bootstrap_ci_median(col, rng, n_resamples)

    return xs, med, lo, hi


def plot_benchmark(ax, benchmark, runs_by_method, n_resamples):
    titles = {"harris": "Harris", "mttkrp": "MTTKRP", "spmv": "SPMV"}
    ax.set_title(titles.get(benchmark, benchmark))
    ax.set_yscale("log")

    for method in METHODS_ORDER:
        runs = runs_by_method.get(method, [])
        if not runs:
            continue

        xs, ys_aligned = align_runs_by_x(runs)
        xs, med, lo, hi = compute_stats(
            xs,
            ys_aligned,
            n_resamples,
            seed=hash((benchmark, method)) & 0xFFFFFFFF,
        )

        mask = np.isfinite(med) & (med > 0)
        if not np.any(mask):
            continue

        color = METHOD_COLORS[method]
        ax.plot(xs[mask], med[mask], label=METHOD_LABELS[method], color=color, linewidth=2)

        if lo is not None:
            band = mask & np.isfinite(lo) & np.isfinite(hi) & (lo > 0) & (hi > 0)
            if np.any(band):
                ax.fill_between(xs[band], lo[band], hi[band], color=color, alpha=0.2)

    ax.set_xlabel("executions")
    ax.set_ylim(top=1e4)
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    ax.legend()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="results_catbench")
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()

    root = Path(args.root)

    data = {}
    for bench in BENCHMARKS:
        data[bench] = collect_runs(discover_files(root, bench))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        "execution time (s)",
        y=1.02,
    )

    for ax in axes:
        ax.set_ylabel("execution time (s)")

    for ax, bench in zip(axes, BENCHMARKS):
        plot_benchmark(ax, bench, data.get(bench, {}), args.bootstrap)

    fig.tight_layout()
    fig.savefig("performance_evolution_catbench.png", dpi=200, bbox_inches="tight")
    fig.savefig("performance_evolution_catbench.pdf", bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()