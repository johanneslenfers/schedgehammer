# Dockerfile for reproducing performance distribution analysis results
FROM python:3.10-slim

# Install system dependencies required for TVM, TACO, and compilation
# This layer is cached unless system dependencies change
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    llvm \
    llvm-dev \
    libtinfo-dev \
    zlib1g-dev \
    libedit-dev \
    libxml2-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Build TACO library first (this layer is cached unless TACO source changes)
# Copy only TACO directory to leverage Docker cache (exclude build dir and .git if they exist)
COPY taco/ ./taco/
# Remove .git directory if it exists to prevent CMake from trying to update submodules
# Build TACO - disable git submodule check and tests since we're not in a git repo
# We only need the library, not the tests
RUN cd taco && \
    rm -rf build .git && \
    mkdir -p build && \
    cd build && \
    cmake -DCMAKE_BUILD_TYPE=Release -DGIT_SUBMODULE=OFF -DBUILD_TESTING=OFF .. && \
    make -j$(nproc) taco && \
    cd ../..

# Copy requirements and install Python dependencies
# This layer is cached unless requirements change
COPY requirements.txt pyproject.toml ./
RUN pip install --upgrade pip wheel setuptools && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project and install the package
# This layer changes most frequently (code changes)
COPY . .

# Install the package (this will build taco_bindings which needs libtaco.so)
# Set LD_LIBRARY_PATH so Python can find libtaco.so at runtime
ENV LD_LIBRARY_PATH=/app/taco/build/lib
RUN pip install --no-cache-dir .

# Make the entrypoint scripts executable
RUN chmod +x performance_distribution/run_analysis.sh
RUN chmod +x examples/schedules/tvm/run_mm_benchmark.sh
RUN chmod +x examples/schedules/taco/run_taco_gemm_benchmark.sh

# Set the entrypoint to run the analysis and generate the plot
ENTRYPOINT ["performance_distribution/run_analysis.sh"]
