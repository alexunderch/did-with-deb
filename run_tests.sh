#!/bin/bash

# 1. Add the current root directory to PYTHONPATH so subdirectories can import each other
export PYTHONPATH="${PWD}:${PYTHONPATH}"

echo "========================================="
echo " Discovering and running all the tests"
echo "========================================="

# 2. Run test discovery
# -s .       Starts looking in the current directory
# -p "*_test.py"  Matches your test file naming convention
python3 -m unittest discover -s . -p "*_test.py"

# 3. Capture the overall exit code (important for CI/CD pipelines)
exit $?
