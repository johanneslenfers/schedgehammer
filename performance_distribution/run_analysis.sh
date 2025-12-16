#!/bin/bash
# Entrypoint script to run performance distribution analysis and generate plot

set -e

# Get optional arguments for number of schedules and variants
NUM_SCHEDULES=${NUM_SCHEDULES:-5}
VARIANTS_PER_SCHEDULE=${VARIANTS_PER_SCHEDULE:-2}
BACKEND=${BACKEND:-tvm}

# Use results directory if it exists (mounted volume), otherwise use current directory
RESULTS_DIR="${RESULTS_DIR:-/app}"

# Validate backend choice
if [ "$BACKEND" != "tvm" ] && [ "$BACKEND" != "taco" ]; then
    echo "Error: BACKEND must be either 'tvm' or 'taco' (got: $BACKEND)"
    exit 1
fi

echo "Starting performance distribution analysis..."
echo "Backend: $BACKEND"
echo "Configuration: ${NUM_SCHEDULES} schedules, ${VARIANTS_PER_SCHEDULE} variants per schedule"

# Change to app directory and run the analysis
cd /app
export RESULTS_DIR

if [ "$BACKEND" = "tvm" ]; then
    if ! python performance_distribution/performance_distribution.py \
        --num-schedules "$NUM_SCHEDULES" \
        --variants-per-schedule "$VARIANTS_PER_SCHEDULE"; then
        echo "Error: TVM performance distribution script failed"
        exit 1
    fi
    JSON_FILE="$RESULTS_DIR/performance_distribution_results_mttkrp_tvm.json"
    PLOT_FILE="$RESULTS_DIR/performance_distribution_plot_tvm.png"
else
    if ! python performance_distribution/performance_distribution_taco.py \
        --num-schedules "$NUM_SCHEDULES" \
        --variants-per-schedule "$VARIANTS_PER_SCHEDULE"; then
        echo "Error: TACO performance distribution script failed"
        echo "Check the output above for error messages"
        exit 1
    fi
    JSON_FILE="$RESULTS_DIR/performance_distribution_results_mttkrp_taco.json"
    PLOT_FILE="$RESULTS_DIR/performance_distribution_plot_taco.png"
fi

if [ -f "$JSON_FILE" ]; then
    echo "Generating plot from results..."
    python performance_distribution/plot.py "$JSON_FILE" "$PLOT_FILE"
    echo "Analysis complete! Results saved to:"
    echo "  - JSON: $JSON_FILE"
    echo "  - Plot: $PLOT_FILE"
else
    echo "Error: JSON file $JSON_FILE not found"
    exit 1
fi

