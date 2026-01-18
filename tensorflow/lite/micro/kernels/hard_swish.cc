/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/lite/kernels/internal/reference/hard_swish.h"

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/op_macros.h"
#include "tensorflow/lite/micro/kernels/hard_swish.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_utils.h"

namespace tflite {
namespace {
void* HardSwishInit(TfLiteContext* context, const char* buffer, size_t length) {
  TFLITE_DCHECK(context->AllocatePersistentBuffer != nullptr);
  return context->AllocatePersistentBuffer(context, sizeof(HardSwishParams));
}

void HardSwishInt16(const HardSwishParams& params,
                    const RuntimeShape& input_shape, const int16_t* input_data,
                    const RuntimeShape& output_shape, int16_t* output_data) {
  const int flat_size = MatchingFlatSize(input_shape, output_shape);

  for (int i = 0; i < flat_size; i++) {
    const int32_t input_value =
        static_cast<int32_t>(input_data[i]) - params.input_zero_point;

    // Reluish calculation:
    // We want input_value * reluish_multiplier.
    // params.reluish_multiplier_fixedpoint_int16 is the top 16 bits of the Q31
    // multiplier. Reconstruct Q31 multiplier.
    int32_t reluish_mult_q31 =
        static_cast<int32_t>(params.reluish_multiplier_fixedpoint_int16) << 16;
    int32_t reluish_val_32 = tflite::MultiplyByQuantizedMultiplier(
        input_value, reluish_mult_q31, params.reluish_multiplier_exponent);

    // Saturate reluish_val_32 to int16 range [-32768, 32767].
    // This value corresponds to the range [-3.0, 3.0] in the reluish scale.
    int16_t reluish_value_int16 = static_cast<int16_t>(std::min<int32_t>(
        std::max<int32_t>(reluish_val_32, std::numeric_limits<int16_t>::min()),
        std::numeric_limits<int16_t>::max()));

    // Convert to sigmoid [0, 1] range mapped to [0, 32767].
    reluish_value_int16 = (reluish_value_int16 + (1 << 15)) >> 1;

    // Output calculation:
    // input_value * output_multiplier * reluish_value.
    // First apply output_multiplier (without exponent).
    int32_t output_mult_q31 =
        static_cast<int32_t>(params.output_multiplier_fixedpoint_int16) << 16;
    int32_t input_on_output_scale_32 = tflite::MultiplyByQuantizedMultiplier(
        input_value, output_mult_q31, 0);

    // Apply reluish factor.
    // reluish_value_int16 is in Q15. Convert to Q31 for multiplication.
    int32_t reluish_factor_q31 =
        static_cast<int32_t>(reluish_value_int16) << 16;
    int32_t output_val_32 = tflite::MultiplyByQuantizedMultiplier(
        input_on_output_scale_32, reluish_factor_q31, 0);

    // Apply output_multiplier_exponent.
    if (params.output_multiplier_exponent > 0) {
      // Saturating left shift.
      // We can use MultiplyByQuantizedMultiplier with 1.0 multiplier and shift.
      // 1.0 in Q31 is std::numeric_limits<int32_t>::max().
      output_val_32 = tflite::MultiplyByQuantizedMultiplier(
          output_val_32, std::numeric_limits<int32_t>::max(),
          params.output_multiplier_exponent);
    } else {
      output_val_32 = gemmlowp::RoundingDivideByPOT(
          output_val_32, -params.output_multiplier_exponent);
    }

    output_val_32 += params.output_zero_point;
    // Saturate to int16
    output_val_32 = std::min<int32_t>(
        std::max<int32_t>(output_val_32, std::numeric_limits<int16_t>::min()),
        std::numeric_limits<int16_t>::max());
    output_data[i] = static_cast<int16_t>(output_val_32);
  }
}

TfLiteStatus HardSwishEval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteEvalTensor* input =
      tflite::micro::GetEvalInput(context, node, kHardSwishInputTensor);
  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kHardSwishOutputTensor);
  HardSwishParams* params = static_cast<HardSwishParams*>(node->user_data);

  switch (input->type) {
    case kTfLiteFloat32: {
      tflite::reference_ops::HardSwish<float>(
          tflite::micro::GetTensorShape(input),
          tflite::micro::GetTensorData<float>(input),
          tflite::micro::GetTensorShape(output),
          tflite::micro::GetTensorData<float>(output));
    } break;
    case kTfLiteInt8: {
      tflite::reference_ops::HardSwish<int8_t>(
          *params, tflite::micro::GetTensorShape(input),
          tflite::micro::GetTensorData<int8_t>(input),
          tflite::micro::GetTensorShape(output),
          tflite::micro::GetTensorData<int8_t>(output));
    } break;
    case kTfLiteInt16: {
      HardSwishInt16(
          *params, tflite::micro::GetTensorShape(input),
          tflite::micro::GetTensorData<int16_t>(input),
          tflite::micro::GetTensorShape(output),
          tflite::micro::GetTensorData<int16_t>(output));
    } break;
    default: {
      MicroPrintf("Unsupported type %s", TfLiteTypeGetName(input->type));
      return kTfLiteError;
    }
  }
  return kTfLiteOk;
}

}  // namespace

TFLMRegistration Register_HARD_SWISH() {
  return tflite::micro::RegisterOp(HardSwishInit, tflite::HardSwishPrepare,
                                   HardSwishEval);
}

}  // namespace tflite
