#!/bin/bash

# Function to build the package
build_package() {
  echo "Building the package and installing dependencies..."
  uv build
  uv pip install -e .
}
# setup zenml
setup_zenml() {
    echo "Setting up ZenML..."
    export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES

    # Wait for ZenML server to be ready
    echo "Waiting for ZenML server to be ready..."
    max_attempts=30
    attempt=1
    while [ $attempt -le $max_attempts ]; do
        if curl -s http://localhost:8080/health > /dev/null; then
            echo "ZenML server is ready!"
            break
        fi
        echo "Attempt $attempt/$max_attempts: Server not ready yet..."
        sleep 2
        attempt=$((attempt + 1))
    done

    if [ $attempt -gt $max_attempts ]; then
        echo "Error: ZenML server did not become ready in time"
        return 1
    fi

    # Connect to the Docker-hosted ZenML server
    echo "Connecting to ZenML server..."
    if zenml connect --url=http://localhost:8080 --username=default --password=zenml; then
        echo "ZenML connection established successfully"
        zenml status
    else
        echo "Failed to connect to ZenML server"
        return 1
    fi
}

# clean codebase
clean() {
  echo "Cleaning the codebase..."
  find . -type d -name ".ruff_cache" -exec rm -rf {} +
  find . -type d -name ".pytest_cache" -exec rm -rf {} +
  find . -type d -name ".mypy_cache" -exec rm -rf {} +
  find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} +
  find . -type d -name "cachedir" -exec rm -rf {} +
  find . -type d -name "*.egg-info" -exec rm -rf {} +
  echo "Codebase cleaned."
}

# For CI: Run specific functions if arguments are provided
if [ $# -gt 0 ]; then
    for func in "$@"; do
        $func
    done
    exit 0
fi