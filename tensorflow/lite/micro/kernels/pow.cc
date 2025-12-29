/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/kernels/internal/reference/pow.h"

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/micro_log.h"

namespace tflite {

namespace {

// Input/output tensor index.
constexpr int kInputTensor1 = 0;
constexpr int kInputTensor2 = 1;
constexpr int kOutputTensor = 0;

// Op data for pow op.
struct OpData {
  bool requires_broadcast;
};

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  TFLITE_DCHECK(context->AllocatePersistentBuffer != nullptr);
  return context->AllocatePersistentBuffer(context, sizeof(OpData));
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 2);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  OpData* data = reinterpret_cast<OpData*>(node->user_data);

  MicroContext* micro_context = GetMicroContext(context);

  TfLiteTensor* input1 = micro_context->AllocateTempInputTensor(node, kInputTensor1);
  TF_LITE_ENSURE(context, input1 != nullptr);
  TfLiteTensor* input2 = micro_context->AllocateTempInputTensor(node, kInputTensor2);
  TF_LITE_ENSURE(context, input2 != nullptr);
  TfLiteTensor* output = micro_context->AllocateTempOutputTensor(node, kOutputTensor);
  TF_LITE_ENSURE(context, output != nullptr);

  TF_LITE_ENSURE_TYPES_EQ(context, input1->type, input2->type);

  const TfLiteType type = input1->type;
  if (type != kTfLiteInt32 && type != kTfLiteFloat32) {
    MicroPrintf("Unsupported data type %s.", TfLiteTypeGetName(type));

    micro_context->DeallocateTempTfLiteTensor(input1);
    micro_context->DeallocateTempTfLiteTensor(input2);
    micro_context->DeallocateTempTfLiteTensor(output);
    return kTfLiteError;
  }
  output->type = type;

  data->requires_broadcast = !HaveSameShapes(input1, input2);

  micro_context->DeallocateTempTfLiteTensor(input1);
  micro_context->DeallocateTempTfLiteTensor(input2);
  micro_context->DeallocateTempTfLiteTensor(output);
  return kTfLiteOk;
}

template <typename T>
void PowImpl(const TfLiteEvalTensor* input1, const TfLiteEvalTensor* input2,
             TfLiteEvalTensor* output, bool requires_broadcast) {
  if (requires_broadcast) {
    reference_ops::BroadcastPow4D(
        tflite::micro::GetTensorShape(input1),
        tflite::micro::GetTensorData<T>(input1),
        tflite::micro::GetTensorShape(input2),
        tflite::micro::GetTensorData<T>(input2),
        tflite::micro::GetTensorShape(output),
        tflite::micro::GetTensorData<T>(output));
  } else {
    reference_ops::Pow(tflite::micro::GetTensorShape(input1),
                       tflite::micro::GetTensorData<T>(input1),
                       tflite::micro::GetTensorShape(input2),
                       tflite::micro::GetTensorData<T>(input2),
                       tflite::micro::GetTensorShape(output),
                       tflite::micro::GetTensorData<T>(output));
  }
}

TfLiteStatus CheckValue(TfLiteContext* context,
                        const TfLiteEvalTensor* input) {
  const int64_t num_elements = tflite::micro::GetTensorShape(input).FlatSize();
  const int32_t* data = tflite::micro::GetTensorData<int32_t>(input);
  for (int i = 0; i < num_elements; ++i) {
    if (data[i] < 0) {
      MicroPrintf("POW does not support negative value for int32.");
      return kTfLiteError;
    }
  }
  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  OpData* data = reinterpret_cast<OpData*>(node->user_data);

  const TfLiteEvalTensor* input1 =
      tflite::micro::GetEvalInput(context, node, kInputTensor1);
  const TfLiteEvalTensor* input2 =
      tflite::micro::GetEvalInput(context, node, kInputTensor2);
  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kOutputTensor);

  switch (output->type) {
    case kTfLiteInt32: {
      // TensorFlow does not support negative for int32.
      TF_LITE_ENSURE_OK(context, CheckValue(context, input2));
      PowImpl<int32_t>(input1, input2, output, data->requires_broadcast);
      break;
    }
    case kTfLiteFloat32: {
      PowImpl<float>(input1, input2, output, data->requires_broadcast);
      break;
    }
    default: {
      MicroPrintf("Unsupported data type: %d", output->type);
      return kTfLiteError;
    }
  }
  return kTfLiteOk;
}

}  // namespace

TFLMRegistration Register_POW() {
  return tflite::micro::RegisterOp(Init, Prepare, Eval);
}

}  // namespace tflite
