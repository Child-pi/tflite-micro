#!/bin/bash
set -e

# Define paths
KERNEL_DIR="tensorflow/lite/micro/kernels"
MY_TEST_DIR="tensorflow/lite/micro/kernels/my_test"
MAKEFILE="tensorflow/lite/micro/tools/make/Makefile"

# Backup the original add.cc
if [ -f "${KERNEL_DIR}/add.cc" ]; then
  cp "${KERNEL_DIR}/add.cc" "${KERNEL_DIR}/add.cc.bak"
else
  echo "Error: ${KERNEL_DIR}/add.cc not found!"
  exit 1
fi

# Replace with the optimized implementation
echo "Copying optimized add.cc from ${MY_TEST_DIR}..."
cp "${MY_TEST_DIR}/add.cc" "${KERNEL_DIR}/add.cc"

# Build the test
echo "Building kernel_add_test..."
make -f "${MAKEFILE}" kernel_add_test

# Restore the original add.cc
echo "Restoring original add.cc..."
mv "${KERNEL_DIR}/add.cc.bak" "${KERNEL_DIR}/add.cc"

echo "Build complete."
