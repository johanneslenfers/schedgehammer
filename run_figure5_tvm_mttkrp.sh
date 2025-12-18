#!/bin/bash
# Script to run MTTKRP benchmark in Docker and show where to find results

set -e

# Configuration
IMAGE_NAME="schedgehammer-cc-artifact"
CONTAINER_NAME="schedgehammer-mttkrp-benchmark-$$"
RESULTS_DIR="$(pwd)/results"
ITERATIONS=${ITERATIONS:-63}
RUNS=${RUNS:-5}
BUDGET=${BUDGET:-100}

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "Interrupted! Stopping Docker container..."
    docker stop "$CONTAINER_NAME" >/dev/null 2>&1 || true
    docker rm "$CONTAINER_NAME" >/dev/null 2>&1 || true
    exit 130
}

# Trap SIGINT and SIGTERM to cleanup
trap cleanup SIGINT SIGTERM

echo "=========================================="
echo "Schedgehammer MTTKRP Benchmark"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  - Ansor iterations: ${ITERATIONS}"
echo "  - Runs per tuner: ${RUNS} (applies to genetic_tuner, random_tuner, and ansor)"
echo "  - Schedgehammer budget per run: ${BUDGET}"
echo ""

# Create results directory if it doesn't exist
mkdir -p "$RESULTS_DIR"

# Always rebuild the image to ensure latest code is used
echo "Building Docker image '$IMAGE_NAME'..."
docker build -t "$IMAGE_NAME" .
echo ""

echo "Running MTTKRP benchmark analysis..."
echo "This may take a while depending on the number of iterations..."
echo "Press Ctrl+C to stop..."
echo ""

# Run the container with volume mount and override entrypoint
# Use --name to be able to stop it, -it for interactive TTY to allow signal forwarding and real-time output
# PYTHONUNBUFFERED=1 ensures Python output is not buffered and appears immediately
set +e  # Temporarily disable exit on error for docker run
docker run -it --name "$CONTAINER_NAME" \
    -e ITERATIONS="$ITERATIONS" \
    -e RUNS="$RUNS" \
    -e BUDGET="$BUDGET" \
    -e RESULTS_DIR="/app/results" \
    -e PYTHONUNBUFFERED=1 \
    -v "$RESULTS_DIR:/app/results" \
    --entrypoint examples/schedules/tvm/run_mttkrp_benchmark.sh \
    "$IMAGE_NAME"
EXIT_CODE=$?
set -e  # Re-enable exit on error

# Cleanup container (whether successful or not)
docker rm "$CONTAINER_NAME" >/dev/null 2>&1 || true

# Remove trap on completion
trap - SIGINT SIGTERM

# Exit with the docker run exit code
if [ $EXIT_CODE -ne 0 ]; then
    exit $EXIT_CODE
fi

echo ""
echo "=========================================="
echo "Analysis Complete!"
echo "=========================================="
echo ""

PLOT_FILE="$RESULTS_DIR/Figure5_6_mttkrp.png"

echo "Results are available at:"
echo "  📊 Plot: $PLOT_FILE"
echo ""
echo "To view the plot, open:"
echo "  $PLOT_FILE"
echo ""
echo ""
echo "Usage: ./run_figure5_tvm_mttkrp.sh"
echo ""
echo "Optional environment variables:"
echo "  ITERATIONS - Number of ansor iterations (default: 63)"
echo "  RUNS - Number of runs per tuner (applies to genetic_tuner, random_tuner, and ansor) (default: 5)"
echo "  BUDGET - Schedgehammer evaluation budget per run (default: 100)"
echo ""

