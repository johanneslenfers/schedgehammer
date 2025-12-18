#!/bin/bash
set -e

# Get optional arguments for iterations, runs, and budget
ITERATIONS=${ITERATIONS:-63}
RUNS=${RUNS:-1}
BUDGET=${BUDGET:-100}

# Use results directory if it exists (mounted volume), otherwise use current directory
RESULTS_DIR="${RESULTS_DIR:-/app/results}"

echo "Figure 4"

# Change to app directory
cd /app
export RESULTS_DIR

# Create necessary directories
mkdir -p "$RESULTS_DIR/catbench"

python benchmark/run_benchmark.py

python benchmark/pyatf_mttkrp.py
python benchmark/pyatf_spmv.py
python benchmark/pyatf_harris.py

python benchmark/opentuner_mttkrp.py
python benchmark/opentuner_spmv.py
python benchmark/opentuner_harris.py

python plot_catbench.py