#!/bin/bash
# Locate the binary. The path includes _cmsis_nn because of the OPTIMIZED_KERNEL_DIR flag.
# We need to find where it is generated.
BINARY=$(find gen -name kernel_add_test | grep cmsis_nn | head -n 1)

if [ -z "$BINARY" ]; then
  echo "Binary not found. Build might have failed."
  exit 1
fi

echo "Running $BINARY"
$BINARY
