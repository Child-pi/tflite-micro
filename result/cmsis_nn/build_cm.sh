#!/bin/bash
# Build with CMSIS-NN optimized kernels
# This assumes we are building on host (x86) but using the CMSIS-NN sources.
# Note: CMSIS-NN is typically for Cortex-M. Building on x86 might fail if it relies on ARM assembly or specific libs not available for x86.
# However, the Makefile seems to support it or at least attempts to use the sources.
# If this fails due to architecture mismatch, it's expected without a cross-compiler.

make -f tensorflow/lite/micro/tools/make/Makefile OPTIMIZED_KERNEL_DIR=cmsis_nn kernel_add_test
