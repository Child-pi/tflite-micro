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
#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/kernels/kernel_runner.h"
#include "tensorflow/lite/micro/test_helpers.h"
#include "tensorflow/lite/micro/testing/micro_test.h"

namespace tflite {

extern TFLMRegistration Register_POW();

namespace testing {
namespace {

void ExecutePowTest(std::initializer_list<int> input1_shape,
                    std::initializer_list<int32_t> input1_data,
                    std::initializer_list<int> input2_shape,
                    std::initializer_list<int32_t> input2_data,
                    std::initializer_list<int> output_shape,
                    std::initializer_list<int32_t> expected_output_data) {
  int* input1_dims_data = new int[input1_shape.size() + 1];
  input1_dims_data[0] = input1_shape.size();
  std::copy(input1_shape.begin(), input1_shape.end(), input1_dims_data + 1);
  TfLiteIntArray* input1_dims = IntArrayFromInts(input1_dims_data);

  int* input2_dims_data = new int[input2_shape.size() + 1];
  input2_dims_data[0] = input2_shape.size();
  std::copy(input2_shape.begin(), input2_shape.end(), input2_dims_data + 1);
  TfLiteIntArray* input2_dims = IntArrayFromInts(input2_dims_data);

  int* output_dims_data = new int[output_shape.size() + 1];
  output_dims_data[0] = output_shape.size();
  std::copy(output_shape.begin(), output_shape.end(), output_dims_data + 1);
  TfLiteIntArray* output_dims = IntArrayFromInts(output_dims_data);

  const int output_elements = ElementCount(*output_dims);
  int32_t* output_data = new int32_t[output_elements];

  constexpr int kInputCount = 2;
  constexpr int kOutputCount = 1;
  TfLiteTensor tensors[kInputCount + kOutputCount];

  // Use arrays for CreateTensor
  int32_t* input1_array = new int32_t[input1_data.size()];
  std::copy(input1_data.begin(), input1_data.end(), input1_array);

  int32_t* input2_array = new int32_t[input2_data.size()];
  std::copy(input2_data.begin(), input2_data.end(), input2_array);

  tensors[0] = CreateTensor(input1_array, input1_dims);
  tensors[1] = CreateTensor(input2_array, input2_dims);
  tensors[2] = CreateTensor(output_data, output_dims);

  int inputs_array_data[] = {2, 0, 1};
  TfLiteIntArray* inputs_array = IntArrayFromInts(inputs_array_data);
  int outputs_array_data[] = {1, 2};
  TfLiteIntArray* outputs_array = IntArrayFromInts(outputs_array_data);

  const TFLMRegistration registration = tflite::Register_POW();
  micro::KernelRunner runner(registration, tensors, kInputCount + kOutputCount,
                             inputs_array, outputs_array,
                             /*builtin_data=*/nullptr);

  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, runner.InitAndPrepare());
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, runner.Invoke());

  for (int i = 0; i < output_elements; ++i) {
    TF_LITE_MICRO_EXPECT_EQ(expected_output_data.begin()[i], output_data[i]);
  }

  delete[] output_data;
  delete[] input1_array;
  delete[] input2_array;
  delete[] input1_dims_data;
  delete[] input2_dims_data;
  delete[] output_dims_data;
}

void ExecutePowTestFloat(std::initializer_list<int> input1_shape,
                         std::initializer_list<float> input1_data,
                         std::initializer_list<int> input2_shape,
                         std::initializer_list<float> input2_data,
                         std::initializer_list<int> output_shape,
                         std::initializer_list<float> expected_output_data) {
  int* input1_dims_data = new int[input1_shape.size() + 1];
  input1_dims_data[0] = input1_shape.size();
  std::copy(input1_shape.begin(), input1_shape.end(), input1_dims_data + 1);
  TfLiteIntArray* input1_dims = IntArrayFromInts(input1_dims_data);

  int* input2_dims_data = new int[input2_shape.size() + 1];
  input2_dims_data[0] = input2_shape.size();
  std::copy(input2_shape.begin(), input2_shape.end(), input2_dims_data + 1);
  TfLiteIntArray* input2_dims = IntArrayFromInts(input2_dims_data);

  int* output_dims_data = new int[output_shape.size() + 1];
  output_dims_data[0] = output_shape.size();
  std::copy(output_shape.begin(), output_shape.end(), output_dims_data + 1);
  TfLiteIntArray* output_dims = IntArrayFromInts(output_dims_data);

  const int output_elements = ElementCount(*output_dims);
  float* output_data = new float[output_elements];

  constexpr int kInputCount = 2;
  constexpr int kOutputCount = 1;
  TfLiteTensor tensors[kInputCount + kOutputCount];

  float* input1_array = new float[input1_data.size()];
  std::copy(input1_data.begin(), input1_data.end(), input1_array);

  float* input2_array = new float[input2_data.size()];
  std::copy(input2_data.begin(), input2_data.end(), input2_array);

  tensors[0] = CreateTensor(input1_array, input1_dims);
  tensors[1] = CreateTensor(input2_array, input2_dims);
  tensors[2] = CreateTensor(output_data, output_dims);

  int inputs_array_data[] = {2, 0, 1};
  TfLiteIntArray* inputs_array = IntArrayFromInts(inputs_array_data);
  int outputs_array_data[] = {1, 2};
  TfLiteIntArray* outputs_array = IntArrayFromInts(outputs_array_data);

  const TFLMRegistration registration = tflite::Register_POW();
  micro::KernelRunner runner(registration, tensors, kInputCount + kOutputCount,
                             inputs_array, outputs_array,
                             /*builtin_data=*/nullptr);

  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, runner.InitAndPrepare());
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, runner.Invoke());

  for (int i = 0; i < output_elements; ++i) {
    TF_LITE_MICRO_EXPECT_NEAR(expected_output_data.begin()[i], output_data[i],
                              1e-3f);
  }

  delete[] output_data;
  delete[] input1_array;
  delete[] input2_array;
  delete[] input1_dims_data;
  delete[] input2_dims_data;
  delete[] output_dims_data;
}

}  // namespace
}  // namespace testing
}  // namespace tflite

TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(PowSimpleInt32) {
  tflite::testing::ExecutePowTest(
      /*input1_shape=*/{1, 2, 2, 1},
      /*input1_data=*/{12, 2, 7, 8},
      /*input2_shape=*/{1, 2, 2, 1},
      /*input2_data=*/{1, 2, 3, 1},
      /*output_shape=*/{1, 2, 2, 1},
      /*expected_output_data=*/{12, 4, 343, 8});
}

TF_LITE_MICRO_TEST(PowNegativeAndZeroValueInt32) {
  tflite::testing::ExecutePowTest(
      /*input1_shape=*/{1, 2, 2, 1},
      /*input1_data=*/{0, 2, -7, 8},
      /*input2_shape=*/{1, 2, 2, 1},
      /*input2_data=*/{1, 2, 3, 0},
      /*output_shape=*/{1, 2, 2, 1},
      /*expected_output_data=*/{0, 4, -343, 1});
}

TF_LITE_MICRO_TEST(PowSimpleFloat) {
  tflite::testing::ExecutePowTestFloat(
      /*input1_shape=*/{1, 2, 2, 1},
      /*input1_data=*/{0.3f, 0.4f, 0.7f, 5.8f},
      /*input2_shape=*/{1, 2, 2, 1},
      /*input2_data=*/{0.5f, 2.7f, 3.1f, 3.2f},
      /*output_shape=*/{1, 2, 2, 1},
      /*expected_output_data=*/{0.5477226f, 0.08424846f, 0.33098164f, 277.313f});
}

TF_LITE_MICRO_TEST(PowNegativeFloat) {
  tflite::testing::ExecutePowTestFloat(
      /*input1_shape=*/{1, 2, 2, 1},
      /*input1_data=*/{0.3f, 0.4f, 0.7f, 5.8f},
      /*input2_shape=*/{1, 2, 2, 1},
      /*input2_data=*/{0.5f, -2.7f, 3.1f, -3.2f},
      /*output_shape=*/{1, 2, 2, 1},
      /*expected_output_data=*/{0.5477226f, 11.869653f, 0.33098164f, 0.003606f});
}

TF_LITE_MICRO_TEST(PowBroadcastInt32) {
  tflite::testing::ExecutePowTest(
      /*input1_shape=*/{1, 2, 2, 1},
      /*input1_data=*/{12, 2, 7, 8},
      /*input2_shape=*/{1},
      /*input2_data=*/{4},
      /*output_shape=*/{1, 2, 2, 1},
      /*expected_output_data=*/{20736, 16, 2401, 4096});
}

TF_LITE_MICRO_TEST(PowBroadcastFloat) {
  tflite::testing::ExecutePowTestFloat(
      /*input1_shape=*/{1, 2, 2, 1},
      /*input1_data=*/{12.0f, 2.0f, 7.0f, 8.0f},
      /*input2_shape=*/{1},
      /*input2_data=*/{4.0f},
      /*output_shape=*/{1, 2, 2, 1},
      /*expected_output_data=*/{20736.0f, 16.0f, 2401.0f, 4096.0f});
}

TF_LITE_MICRO_TESTS_END
