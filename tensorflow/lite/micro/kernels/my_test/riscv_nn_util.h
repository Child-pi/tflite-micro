#ifndef TENSORFLOW_LITE_MICRO_KERNELS_MY_TEST_RISCV_NN_UTIL_H_
#define TENSORFLOW_LITE_MICRO_KERNELS_MY_TEST_RISCV_NN_UTIL_H_

#include <stdint.h>
#include <algorithm>
#include "tensorflow/lite/kernels/internal/quantization_util.h"

// Macros expected by the riscv code if not already defined
#ifndef MAX
#define MAX(A,B) ((A) > (B) ? (A) : (B))
#endif
#ifndef MIN
#define MIN(A,B) ((A) < (B) ? (A) : (B))
#endif

namespace tflite {
namespace micro {
namespace riscv {

// Stub for riscv_nn_requantize_ns
// Based on usage: riscv_nn_requantize_ns(val, scale, shift) where shift < 0 is right shift.
// In the original library, this function performs requantization.
// We map it to TFLite's MultiplyByQuantizedMultiplier for correctness in this test environment.
static inline int32_t riscv_nn_requantize_ns(const int32_t val, const int32_t multiplier, const int32_t shift) {
  return tflite::MultiplyByQuantizedMultiplier(val, multiplier, shift);
}

// Function from https://github.com/andestech/libnn/blob/ans-v1_1_1-release/Source/BasicFunctions/riscv_nn_ew_add_s8_asym.c
// Adjusted to be an inline C++ function.
static inline int32_t riscv_nn_ew_add_s8_asym(const int8_t * in_vec1,
                                const int8_t * in_vec2,
                                const int32_t in_offset1,
                                const int32_t in_scale1,
                                const int32_t in_rshift1,
                                const int32_t in_offset2,
                                const int32_t in_scale2,
                                const int32_t in_rshift2,
                                const int32_t lshift,
                                int8_t * out_vec,
                                const int32_t out_offset,
                                const int32_t out_scale,
                                const int32_t out_rshift,
                                const int32_t act_min,
                                const int32_t act_max,
                                const uint32_t size)
{
    int32_t in1, in2, out;
    uint32_t loop = size;

    while (loop > 0)
    {
        // The original C code:
        // in1 = (*in_vec1++ + in_offset1) << lshift;
        // in2 = (*in_vec2++ + in_offset2) << lshift;
        // in1 = riscv_nn_requantize_ns(in1, in_scale1, -in_rshift1);
        // in2 = riscv_nn_requantize_ns(in2, in_scale2, -in_rshift2);

        // Note: lshift in TFLite (left_shift) is typically 20.
        // in_offset is typically -zero_point.

        in1 = (*in_vec1++ + in_offset1) << lshift;
        in2 = (*in_vec2++ + in_offset2) << lshift;
        in1 = riscv_nn_requantize_ns(in1, in_scale1, -in_rshift1);
        in2 = riscv_nn_requantize_ns(in2, in_scale2, -in_rshift2);

        out = in1 + in2;
        out = riscv_nn_requantize_ns(out, out_scale, -out_rshift);
        out += out_offset;
        out = MAX(out, act_min);
        out = MIN(out, act_max);

        *out_vec++ = (int8_t)out;
        loop--;
    }

    return 0;
}

}  // namespace riscv
}  // namespace micro
}  // namespace tflite

#endif // TENSORFLOW_LITE_MICRO_KERNELS_MY_TEST_RISCV_NN_UTIL_H_
