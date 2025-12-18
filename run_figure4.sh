#!/bin/bash
# Script to run catbench benchmarks in Docker and copy results back

set -e

# Configuration
IMAGE_NAME="schedgehammer-cc-artifact"
CONTAINER_NAME="schedgehammer-catbench-$$"
RESULTS_DIR="$(pwd)/results"
CATBENCH_RESULTS_DIR="$(pwd)/catbench_results"

# Cleanup on exit or interrupt
cleanup() {
    echo ""
    echo "Interrupted! Stopping Docker container..."
    docker stop "$CONTAINER_NAME" >/dev/null 2>&1 || true
    docker rm "$CONTAINER_NAME" >/dev/null 2>&1 || true
    exit 130
}

trap cleanup SIGINT SIGTERM

echo "=========================================="
echo "Schedgehammer Catbench Benchmarks"
echo "=========================================="
echo ""
echo "Results will be written to:"
echo "  - $RESULTS_DIR"
echo "  - $CATBENCH_RESULTS_DIR"
echo ""

# Ensure host result directories exist
mkdir -p "$RESULTS_DIR" "$CATBENCH_RESULTS_DIR"

echo "Building Docker image '$IMAGE_NAME'..."
docker build -t "$IMAGE_NAME" .
echo ""

echo "Running catbench benchmarks inside Docker..."
echo "Press Ctrl+C to stop..."
echo ""

# Run the container with result directories mounted
set +e
docker run -it --name "$CONTAINER_NAME" \
    -e PYTHONUNBUFFERED=1 \
    -v "$RESULTS_DIR:/app/results" \
    -v "$CATBENCH_RESULTS_DIR:/app/catbench_results" \
    --entrypoint /bin/bash \
    "$IMAGE_NAME" -c "cd /app && ./run_catbench.sh"
EXIT_CODE=$?
set -e

# Cleanup container regardless of success
docker rm "$CONTAINER_NAME" >/dev/null 2>&1 || true
trap - SIGINT SIGTERM

if [ $EXIT_CODE -ne 0 ]; then
    exit $EXIT_CODE
fi

echo ""
echo "=========================================="
echo "Catbench Runs Complete!"
echo "=========================================="
echo ""
echo "Results are available at:"
echo "  - $RESULTS_DIR"
echo "  - $CATBENCH_RESULTS_DIR"
echo ""
echo "Usage: ./run_docker_catbench.sh"
echo ""