#!/bin/bash
# Script to run performance distribution analysis in Docker and show where to find results

set -e

# Configuration
IMAGE_NAME="schedgehammer-perf"
RESULTS_DIR="$(pwd)/results"
NUM_SCHEDULES=${NUM_SCHEDULES:-500}
VARIANTS_PER_SCHEDULE=${VARIANTS_PER_SCHEDULE:-50}

# Get backend from command-line argument, default to tvm
BACKEND=${1:-tvm}

# Validate backend choice
if [ "$BACKEND" != "tvm" ] && [ "$BACKEND" != "taco" ]; then
    echo "Error: Backend must be either 'tvm' or 'taco' (got: $BACKEND)"
    echo ""
    echo "Usage: ./run_docker_performance_distribution.sh [tvm|taco]"
    echo ""
    echo "Examples:"
    echo "  ./run_docker_performance_distribution.sh tvm"
    echo "  ./run_docker_performance_distribution.sh taco"
    echo ""
    echo "Optional environment variables:"
    echo "  NUM_SCHEDULES - Number of schedules to evaluate (default: 10)"
    echo "  VARIANTS_PER_SCHEDULE - Number of variants per schedule (default: 2)"
    exit 1
fi

echo "=========================================="
echo "Schedgehammer Performance Distribution"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  - Backend: $BACKEND"
echo "  - Schedules: ${NUM_SCHEDULES}"
echo "  - Variants per schedule: ${VARIANTS_PER_SCHEDULE}"
echo ""

# Create results directory if it doesn't exist
mkdir -p "$RESULTS_DIR"

# Always rebuild the image to ensure latest code is used
echo "Building Docker image '$IMAGE_NAME'..."
docker build -t "$IMAGE_NAME" .
echo ""

echo "Running performance distribution analysis..."
echo "This may take a while depending on the number of schedules..."
echo ""

# Run the container with volume mount
docker run --rm \
    -e NUM_SCHEDULES="$NUM_SCHEDULES" \
    -e VARIANTS_PER_SCHEDULE="$VARIANTS_PER_SCHEDULE" \
    -e BACKEND="$BACKEND" \
    -e RESULTS_DIR="/app/results" \
    -v "$RESULTS_DIR:/app/results" \
    "$IMAGE_NAME"

echo ""
echo "=========================================="
echo "Analysis Complete!"
echo "=========================================="
echo ""

if [ "$BACKEND" = "tvm" ]; then
    PLOT_FILE="$RESULTS_DIR/performance_distribution_plot_tvm.png"
    JSON_FILE="$RESULTS_DIR/performance_distribution_results_mttkrp_tvm.json"
else
    PLOT_FILE="$RESULTS_DIR/performance_distribution_plot_taco.png"
    JSON_FILE="$RESULTS_DIR/performance_distribution_results_mttkrp_taco.json"
fi

echo "Results are available at:"
echo "  📊 Plot: $PLOT_FILE"
echo "  📄 JSON: $JSON_FILE"
echo ""
echo "To view the plot, open:"
echo "  $PLOT_FILE"
echo ""

