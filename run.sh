#!/bin/bash
# Create result directory if it doesn't exist
mkdir -p result

# Locate the compiled binary.
# The path typically follows the pattern: gen/<arch>_default_gcc/bin/kernel_add_test
BINARY=$(find gen -name kernel_add_test -type f | head -n 1)

if [ -z "$BINARY" ]; then
  echo "Error: Test binary 'kernel_add_test' not found."
  echo "Please run ./build.sh first."
  exit 1
fi

echo "Found binary at: $BINARY"
echo "Running test..."

# Run the binary and save output to result/reproduced_result.txt
$BINARY > result/reproduced_result.txt 2>&1

if [ $? -eq 0 ]; then
  echo "Test passed. Results saved to result/reproduced_result.txt"
  cat result/reproduced_result.txt
else
  echo "Test failed."
  cat result/reproduced_result.txt
  exit 1
fi
