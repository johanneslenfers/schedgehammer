#!/bin/bash
# Entrypoint script to run MM benchmark (schedgehammer, ansor, baseline) and generate plot

set -e

# Get optional arguments for iterations, runs, and budget
ITERATIONS=${ITERATIONS:-63}
RUNS=${RUNS:-1}
BUDGET=${BUDGET:-100}

# Use results directory if it exists (mounted volume), otherwise use current directory
RESULTS_DIR="${RESULTS_DIR:-/app/results}"

echo "Starting MM benchmark analysis..."
echo "Configuration: ${ITERATIONS} ansor iterations, ${RUNS} runs per tuner, ${BUDGET} schedgehammer budget"

# Change to app directory
cd /app
export RESULTS_DIR

# Create necessary directories
mkdir -p "$RESULTS_DIR/ansor"
mkdir -p "$RESULTS_DIR/base"
mkdir -p "$RESULTS_DIR/tvm/mm"

# Run schedgehammer benchmark
echo ""
echo "Running schedgehammer benchmark..."
python -u examples/schedules/tvm/tvm_run.py mm

# Run ansor
echo ""
echo "Running ansor..."
python -u examples/schedules/tvm/tvm_run.py mm ansor

# Run baseline
echo ""
echo "Running baseline..."
python -u examples/schedules/tvm/tvm_run.py mm baseline

# Generate plot using existing ArchivedResult.plot() and extend it
echo ""
echo "Generating plot from results..."
python -c "
import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from schedgehammer.benchmark import ArchivedResult

results_dir = Path(os.environ.get('RESULTS_DIR', 'results'))


# Set up plot style
plt.style.use('fivethirtyeight')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['figure.dpi'] = 300
plt.figure(figsize=(10, 6))

# Load schedgehammer results
schedgehammer_dir = results_dir / 'tvm' / 'mm' / 'runs'
archived_res = ArchivedResult()
archived_res.load_runs(str(schedgehammer_dir), ['genetic_tuner', 'random_tuner'])

# Color mapping matching the reference plot
colors = {
    'genetic_tuner': '#1f77b4',  # blue
    'random_tuner': '#ff7f0e',   # orange-red
}

# Plot schedgehammer results
for tuner, runs in archived_res.runs_of_tuners.items():
    if not runs:
        continue
    best_scores_list = ArchivedResult._get_best_scores(runs)
    zipped = list(zip(*best_scores_list)) if best_scores_list and best_scores_list[0] else []
    median = []
    upper_bound = []
    lower_bound = []
    xs = []
    for i in range(len(zipped)):
        xs.append(i)
        median.append(np.median(zipped[i]))
        # Use same percentile calculation as ArchivedResult.plot() for consistency
        # 68.27% confidence interval (approximately 1 standard deviation)
        lower_bound.append(np.percentile(zipped[i], 50 - 68.27 / 2))
        upper_bound.append(np.percentile(zipped[i], 50 + 68.27 / 2))
    
    # Map tuner name to display name
    display_name = 'Schedge. schedule genetic' if tuner == 'genetic_tuner' else 'Schedge. schedule random'
    color = colors.get(tuner, None)
    plt.plot(xs, median, label=display_name, color=color, linewidth=2)
    plt.fill_between(xs, lower_bound, upper_bound, alpha=0.3, color=color)

# Process and plot Ansor results as progression
ansor_file = results_dir / 'ansor' / 'mm.json'
if ansor_file.exists():
    with open(ansor_file, 'r') as f:
        ansor_data = json.load(f)
        if ansor_data and len(ansor_data) > 0:
            # ansor_data is a list of runs, each run is a list of costs per iteration
            # Calculate best-so-far for each run
            ansor_best_scores_list = []
            for run in ansor_data:
                best_scores = []
                best_score = float('inf')
                for cost in run:
                    if cost < best_score:
                        best_score = cost
                    best_scores.append(best_score)
                ansor_best_scores_list.append(best_scores)
            
            # Calculate statistics across runs
            if ansor_best_scores_list:
                max_len = max(len(bs) for bs in ansor_best_scores_list)
                median = []
                lower_bound = []
                upper_bound = []
                xs = []
                for i in range(max_len):
                    # Get values at iteration i across all runs, using last value if run is shorter
                    values = [bs[i] if i < len(bs) else bs[-1] for bs in ansor_best_scores_list]
                    xs.append(i)
                    median.append(np.median(values))
                    # Use same percentile calculation as ArchivedResult.plot() for consistency
                    # 68.27% confidence interval (approximately 1 standard deviation)
                    lower_bound.append(np.percentile(values, 50 - 68.27 / 2))
                    upper_bound.append(np.percentile(values, 50 + 68.27 / 2))
                
                # Plot Ansor with goldenrod color
                plt.plot(xs, median, label='Ansor', color='#daa520', linewidth=2)  # goldenrod
                plt.fill_between(xs, lower_bound, upper_bound, alpha=0.3, color='#daa520')
                
                # Mark the end of Ansor line with an X
                if len(xs) > 0 and len(median) > 0:
                    plt.plot(xs[-1], median[-1], marker='X', markersize=10, 
                            color='#daa520', markeredgecolor='black', markeredgewidth=1,
                            zorder=5, label=None)

# Add baseline as horizontal dashed line
baseline_file = results_dir / 'base' / 'mm.json'
if baseline_file.exists():
    with open(baseline_file, 'r') as f:
        baseline_time = float(json.load(f))
        plt.axhline(y=baseline_time, color='black', linestyle='--', linewidth=2, label='Baseline')

plt.xlabel('executions', fontsize=12)
plt.ylabel('execution time (s)', fontsize=12)
plt.yscale('log')
plt.legend(fontsize=10, loc='upper right')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(str(results_dir / 'mm_benchmark_plot.png'), dpi=300, bbox_inches='tight')
print(f'Plot saved to {results_dir / \"mm_benchmark_plot.png\"}')
"

echo ""
echo "Analysis complete! Results saved to:"
echo "  - Plot: $RESULTS_DIR/Figure5_1.png"

