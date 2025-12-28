/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include <cstdint>

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/kernels/kernel_runner.h"
#include "tensorflow/lite/micro/test_helpers.h"
#include "tensorflow/lite/micro/testing/micro_test.h"

namespace tflite {
namespace testing {
namespace {

// Shapes and values for mixed broadcast tests.
constexpr int kBroadcastOutputDimsCount = 36;
constexpr int kBroadcastNumShapes = 4;

int broadcast_input1_shape[] = {4, 2, 3, 1, 2};
const float broadcast_input1_values[] = {-0.3, 2.3, 0.9,  0.5, 0.8, -1.1,
                                         1.2,  2.8, -1.6, 0.0, 0.7, -2.2};
const float broadcast_input2_values[] = {0.2, 0.3, -0.4, 0.5, 1.0, 0.9};
// clang-format off
const float
    broadcast_goldens[kBroadcastNumShapes][kBroadcastOutputDimsCount] = {
        {-0.1, 2.6,  -0.7, 2.8,  0.7,  3.2,  1.1, 0.8,  0.5, 1.0,  1.9, 1.4,
         1.0,  -0.8, 0.4,  -0.6, 1.8,  -0.2, 1.4, 3.1,  0.8, 3.3,  2.2, 3.7,
         -1.4, 0.3,  -2.0, 0.5,  -0.6, 0.9,  0.9, -1.9, 0.3, -1.7, 1.7, -1.3},
        {-0.1, 2.6, 0.5, 1.0, 1.8, -0.2, 1.4, 3.1, -2.0, 0.5, 1.7, -1.3},
        {-0.1, 2.5,  0.0,  2.6,  -0.7, 1.9,  1.1, 0.7,  1.2, 0.8,  0.5, 0.1,
         1.0,  -0.9, 1.1,  -0.8, 0.4,  -1.5, 1.7, 3.3,  2.2, 3.8,  2.1, 3.7,
         -1.1, 0.5,  -0.6, 1.0,  -0.7, 0.9,  1.2, -1.7, 1.7, -1.2, 1.6, -1.3},
        {-0.1, 2.5, 1.2, 0.8, 0.4, -1.5, 1.7, 3.3, -0.6, 1.0, 1.6, -1.3},
};
// clang-format on

constexpr int kBroadcastMaxShapeSize = 5;
int broadcast_input2_shapes[kBroadcastNumShapes][kBroadcastMaxShapeSize] = {
    {4, 1, 1, 3, 2},
    {4, 1, 3, 1, 2},
    {4, 2, 1, 3, 1},
    {4, 2, 3, 1, 1},
};
int broadcast_output_shapes[kBroadcastNumShapes][kBroadcastMaxShapeSize] = {
    {4, 2, 3, 3, 2},
    {4, 2, 3, 1, 2},
    {4, 2, 3, 3, 2},
    {4, 2, 3, 1, 2},
};

template <typename T>
void ValidateAddGoldens(TfLiteTensor* tensors, int tensors_size,
                        const T* golden, T* output, int output_size,
                        TfLiteFusedActivation activation,
                        float tolerance = 1e-5) {
  TfLiteAddParams builtin_data;
  builtin_data.activation = activation;

  int inputs_array_data[] = {2, 0, 1};
  TfLiteIntArray* inputs_array = IntArrayFromInts(inputs_array_data);
  int outputs_array_data[] = {1, 2};
  TfLiteIntArray* outputs_array = IntArrayFromInts(outputs_array_data);

  const TFLMRegistration registration = Register_ADD();
  micro::KernelRunner runner(registration, tensors, tensors_size, inputs_array,
                             outputs_array, &builtin_data);

  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, runner.InitAndPrepare());
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, runner.Invoke());

  for (int i = 0; i < output_size; ++i) {
    TF_LITE_MICRO_EXPECT_NEAR(golden[i], output[i], tolerance);
  }
}

void TestAddFloat(int* input1_dims_data, const float* input1_data,
                  int* input2_dims_data, const float* input2_data,
                  int* output_dims_data, const float* expected_output,
                  TfLiteFusedActivation activation, float* output_data) {
  TfLiteIntArray* input1_dims = IntArrayFromInts(input1_dims_data);
  TfLiteIntArray* input2_dims = IntArrayFromInts(input2_dims_data);
  TfLiteIntArray* output_dims = IntArrayFromInts(output_dims_data);

  constexpr int inputs_size = 2;
  constexpr int outputs_size = 1;
  constexpr int tensors_size = inputs_size + outputs_size;
  TfLiteTensor tensors[tensors_size] = {
      CreateTensor(input1_data, input1_dims),
      CreateTensor(input2_data, input2_dims),
      CreateTensor(output_data, output_dims),
  };

  ValidateAddGoldens(tensors, tensors_size, expected_output, output_data,
                     ElementCount(*output_dims), activation);
}

void TestAddInt32(int* input1_dims_data, const int32_t* input1_data,
                  int* input2_dims_data, const int32_t* input2_data,
                  int* output_dims_data, const int32_t* expected_output,
                  TfLiteFusedActivation activation, int32_t* output_data) {
  TfLiteIntArray* input1_dims = IntArrayFromInts(input1_dims_data);
  TfLiteIntArray* input2_dims = IntArrayFromInts(input2_dims_data);
  TfLiteIntArray* output_dims = IntArrayFromInts(output_dims_data);

  constexpr int inputs_size = 2;
  constexpr int outputs_size = 1;
  constexpr int tensors_size = inputs_size + outputs_size;
  TfLiteTensor tensors[tensors_size] = {
      CreateTensor(input1_data, input1_dims),
      CreateTensor(input2_data, input2_dims),
      CreateTensor(output_data, output_dims),
  };
  ValidateAddGoldens(tensors, tensors_size, expected_output, output_data,
                     ElementCount(*output_dims), activation);
}

template <typename T>
void TestAddQuantized(int* input1_dims_data, const float* input1_data,
                      T* input1_quantized, float input1_scale,
                      int input1_zero_point, int* input2_dims_data,
                      const float* input2_data, T* input2_quantized,
                      float input2_scale, int input2_zero_point,
                      int* output_dims_data, const float* golden,
                      T* golden_quantized, float output_scale,
                      int output_zero_point, TfLiteFusedActivation activation,
                      T* output_data) {
  TfLiteIntArray* input1_dims = IntArrayFromInts(input1_dims_data);
  TfLiteIntArray* input2_dims = IntArrayFromInts(input2_dims_data);
  TfLiteIntArray* output_dims = IntArrayFromInts(output_dims_data);

  constexpr int inputs_size = 2;
  constexpr int outputs_size = 1;
  constexpr int tensors_size = inputs_size + outputs_size;
  TfLiteTensor tensors[tensors_size] = {
      tflite::testing::CreateQuantizedTensor(input1_data, input1_quantized,
                                             input1_dims, input1_scale,
                                             input1_zero_point),
      tflite::testing::CreateQuantizedTensor(input2_data, input2_quantized,
                                             input2_dims, input2_scale,
                                             input2_zero_point),
      tflite::testing::CreateQuantizedTensor(output_data, output_dims,
                                             output_scale, output_zero_point),
  };
  tflite::Quantize(golden, golden_quantized, ElementCount(*output_dims),
                   output_scale, output_zero_point);

  ValidateAddGoldens(tensors, tensors_size, golden_quantized, output_data,
                     ElementCount(*output_dims), activation);
}

}  // namespace
}  // namespace testing
}  // namespace tflite

TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(FloatAddNoActivation) {
  int inout_shape[] = {4, 1, 2, 2, 1};
  const float input1_values[] = {-2.0, 0.2, 0.7, 0.8};
  const float input2_values[] = {0.1, 0.2, 0.3, 0.5};
  const float golden_values[] = {-1.9, 0.4, 1.0, 1.3};
  constexpr int kOutputDimsCount = 4;
  float output_data[kOutputDimsCount];
  tflite::testing::TestAddFloat(inout_shape, input1_values, inout_shape,
                                input2_values, inout_shape, golden_values,
                                kTfLiteActNone, output_data);
}

TF_LITE_MICRO_TEST(FloatAddActivationRelu1) {
  int inout_shape[] = {4, 1, 2, 2, 1};
  const float input1_values[] = {-2.0, 0.2, 0.7, 0.8};
  const float input2_values[] = {0.1, 0.2, 0.3, 0.5};
  const float golden_values[] = {-1.0, 0.4, 1.0, 1.0};

  constexpr int kOutputDimsCount = 4;
  float output_data[kOutputDimsCount];
  tflite::testing::TestAddFloat(inout_shape, input1_values, inout_shape,
                                input2_values, inout_shape, golden_values,
                                kTfLiteActReluN1To1, output_data);
}

TF_LITE_MICRO_TEST(FloatAddActivationRelu) {
  int inout_shape[] = {4, 1, 2, 2, 1};
  const float input1_values[] = {-2.0, 0.2, 0.7, 0.8};
  const float input2_values[] = {0.1, 0.2, 0.3, 0.5};
  const float golden_values[] = {0.0, 0.4, 1.0, 1.3};

  constexpr int kOutputDimsCount = 4;
  float output_data[kOutputDimsCount];
  tflite::testing::TestAddFloat(inout_shape, input1_values, inout_shape,
                                input2_values, inout_shape, golden_values,
                                kTfLiteActRelu, output_data);
}

TF_LITE_MICRO_TEST(FloatAddActivationRelu6) {
  int inout_shape[] = {4, 1, 2, 2, 1};
  const float input1_values[] = {-2.0, 0.2, 7.0, 0.8};
  const float input2_values[] = {0.1, 0.2, 0.3, 5.5};
  const float golden_values[] = {0.0, 0.4, 6.0, 6.0};

  constexpr int kOutputDimsCount = 4;
  float output_data[kOutputDimsCount];
  tflite::testing::TestAddFloat(inout_shape, input1_values, inout_shape,
                                input2_values, inout_shape, golden_values,
                                kTfLiteActRelu6, output_data);
}

TF_LITE_MICRO_TEST(FloatAddVariousInputShapes) {
  constexpr int kOutputDimsCount = 6;
  float output_data[kOutputDimsCount];

  const float input1_values[] = {-2.0, 0.2, 0.7, 0.8, 1.1, 2.0};
  const float input2_values[] = {0.1, 0.2, 0.3, 0.5, 1.1, 0.1};
  const float expected_output[] = {-1.9, 0.4, 1.0, 1.3, 2.2, 2.1};

  constexpr int num_shapes = 4;
  constexpr int max_shape_size = 5;
  int test_shapes[num_shapes][max_shape_size] = {
      {1, 6},
      {2, 2, 3},
      {3, 2, 1, 3},
      {4, 1, 3, 1, 2},
  };

  for (int i = 0; i < num_shapes; ++i) {
    tflite::testing::TestAddFloat(test_shapes[i], input1_values, test_shapes[i],
                                  input2_values, test_shapes[i],
                                  expected_output, kTfLiteActNone, output_data);
  }
}

TF_LITE_MICRO_TEST(FloatAddWithScalarBroadcast) {
  constexpr int kOutputDimsCount = 6;
  float output_data[kOutputDimsCount];

  const float input1_values[] = {-2.0, 0.2, 0.7, 0.8, 1.1, 2.0};
  int input2_shape[] = {0};
  const float input2_values[] = {0.1};
  const float expected_output[] = {-1.9, 0.3, 0.8, 0.9, 1.2, 2.1};

  constexpr int num_shapes = 4;
  constexpr int max_shape_size = 5;
  int test_shapes[num_shapes][max_shape_size] = {
      {1, 6},
      {2, 2, 3},
      {3, 2, 1, 3},
      {4, 1, 3, 1, 2},
  };

  for (int i = 0; i < num_shapes; ++i) {
    tflite::testing::TestAddFloat(test_shapes[i], input1_values, input2_shape,
                                  input2_values, test_shapes[i],
                                  expected_output, kTfLiteActNone, output_data);
  }
}

TF_LITE_MICRO_TEST(Int32AddNoActivation) {
  int inout_shape[] = {4, 1, 2, 2, 1};
  const int32_t input1_values[] = {-2, 2147483646, -1, 1146622854};
  const int32_t input2_values[] = {3, 1, -2147483647, -726978367};
  const int32_t golden_values[] = {1, 2147483647, -2147483648, 419644487};
  constexpr int kOutputDimsCount = 4;
  int32_t output_data[kOutputDimsCount];
  tflite::testing::TestAddInt32(inout_shape, input1_values, inout_shape,
                                input2_values, inout_shape, golden_values,
                                kTfLiteActNone, output_data);
}

TF_LITE_MICRO_TEST(QuantizedAddNoActivationInt8) {
  const float scales[] = {0.25, 0.5, 1.0};
  const int zero_points[] = {-10, 4, 13};
  int inout_shape[] = {4, 1, 2, 2, 1};
  const float input1_values[] = {-2.01, -1.01, -0.01, 0.98};
  const float input2_values[] = {1.01, 1.99, 2.99, 4.02};
  const float golden_values[] = {-1, 1, 3, 5};

  constexpr int kOutputDimsCount = 4;
  int8_t input1_quantized[kOutputDimsCount];
  int8_t input2_quantized[kOutputDimsCount];
  int8_t golden_quantized[kOutputDimsCount];
  int8_t output[kOutputDimsCount];

  tflite::testing::TestAddQuantized(
      inout_shape, input1_values, input1_quantized, scales[0], zero_points[0],
      inout_shape, input2_values, input2_quantized, scales[1], zero_points[1],
      inout_shape, golden_values, golden_quantized, scales[2], zero_points[2],
      kTfLiteActNone, output);
}

TF_LITE_MICRO_TEST(QuantizedAddActivationRelu1Int8) {
  const float scales[] = {0.25, 0.5, 1.0};
  const int zero_points[] = {-10, 4, 13};
  int inout_shape[] = {4, 1, 2, 2, 1};
  const float input1_values[] = {-2.01, -1.01, -0.01, 0.98};
  const float input2_values[] = {1.01, 1.99, 2.99, 4.02};
  const float golden_values[] = {-1, 1, 1, 1};

  constexpr int kOutputDimsCount = 4;
  int8_t input1_quantized[kOutputDimsCount];
  int8_t input2_quantized[kOutputDimsCount];
  int8_t golden_quantized[kOutputDimsCount];
  int8_t output[kOutputDimsCount];

  tflite::testing::TestAddQuantized(
      inout_shape, input1_values, input1_quantized, scales[0], zero_points[0],
      inout_shape, input2_values, input2_quantized, scales[1], zero_points[1],
      inout_shape, golden_values, golden_quantized, scales[2], zero_points[2],
      kTfLiteActReluN1To1, output);
}

TF_LITE_MICRO_TEST(QuantizedAddActivationReluInt8) {
  const float scales[] = {0.1, 0.1, 0.1};
  const int zero_points[] = {0, 0, 0};
  int inout_shape[] = {4, 1, 2, 2, 1};
  const float input1_values[] = {-2.0, 0.2, 0.7, 0.8};
  const float input2_values[] = {0.1, 0.2, 0.3, 0.5};
  const float golden_values[] = {0.0, 0.4, 1.0, 1.3};

  constexpr int kOutputDimsCount = 4;
  int8_t input1_quantized[kOutputDimsCount];
  int8_t input2_quantized[kOutputDimsCount];
  int8_t golden_quantized[kOutputDimsCount];
  int8_t output[kOutputDimsCount];

  tflite::testing::TestAddQuantized(
      inout_shape, input1_values, input1_quantized, scales[0], zero_points[0],
      inout_shape, input2_values, input2_quantized, scales[1], zero_points[1],
      inout_shape, golden_values, golden_quantized, scales[2], zero_points[2],
      kTfLiteActRelu, output);
}

TF_LITE_MICRO_TEST(QuantizedAddVariousInputShapesInt8) {
  const float scales[] = {0.1, 0.05, 0.1};
  const int zero_points[] = {-9, 5, 14};

  constexpr int num_shapes = 4;
  constexpr int max_shape_size = 5;
  int test_shapes[num_shapes][max_shape_size] = {
      {1, 6},
      {2, 2, 3},
      {3, 2, 1, 3},
      {4, 1, 3, 1, 2},
  };

  const float input1_values[] = {-2.0, 0.2, 0.7, 0.8, 1.1, 2.0};
  const float input2_values[] = {0.1, 0.2, 0.3, 0.5, 1.1, 0.1};
  const float golden_values[] = {-1.9, 0.4, 1.0, 1.3, 2.2, 2.1};

  constexpr int kOutputDimsCount = 6;
  int8_t input1_quantized[kOutputDimsCount];
  int8_t input2_quantized[kOutputDimsCount];
  int8_t golden_quantized[kOutputDimsCount];
  int8_t output[kOutputDimsCount];

  for (int i = 0; i < num_shapes; i++) {
    tflite::testing::TestAddQuantized(
        test_shapes[i], input1_values, input1_quantized, scales[0],
        zero_points[0], test_shapes[i], input2_values, input2_quantized,
        scales[1], zero_points[1], test_shapes[i], golden_values,
        golden_quantized, scales[2], zero_points[2], kTfLiteActNone, output);
  }
}

TF_LITE_MICRO_TEST(QuantizedAddWithScalarBroadcastFloat) {
  float output_float[tflite::testing::kBroadcastOutputDimsCount];

  for (int i = 0; i < tflite::testing::kBroadcastNumShapes; ++i) {
    tflite::testing::TestAddFloat(tflite::testing::broadcast_input1_shape,
                                  tflite::testing::broadcast_input1_values,
                                  tflite::testing::broadcast_input2_shapes[i],
                                  tflite::testing::broadcast_input2_values,
                                  tflite::testing::broadcast_output_shapes[i],
                                  tflite::testing::broadcast_goldens[i],
                                  kTfLiteActNone, output_float);
  }
}

TF_LITE_MICRO_TEST(QuantizedAddWithScalarBroadcastInt8) {
  const float input1_values[] = {-2.0, 0.2, 0.7, 0.8, 1.1, 2.0};
  int input2_shape[] = {0};
  const float input2_values[] = {0.1};
  const float golden[] = {-1.9, 0.3, 0.8, 0.9, 1.2, 2.1};

  constexpr int num_shapes = 4;
  constexpr int max_shape_size = 5;
  int test_shapes[num_shapes][max_shape_size] = {
      {1, 6},
      {2, 2, 3},
      {3, 2, 1, 3},
      {4, 1, 3, 1, 2},
  };

  const float scales[] = {0.1, 0.05, 0.05};
  const int zero_points[] = {-8, 4, 12};

  constexpr int kOutputDimsCount = 6;
  int8_t input1_quantized[kOutputDimsCount];
  int8_t input2_quantized[kOutputDimsCount];
  int8_t golden_quantized[kOutputDimsCount];
  int8_t output[kOutputDimsCount];

  for (int i = 0; i < num_shapes; ++i) {
    tflite::testing::TestAddQuantized(
        test_shapes[i], input1_values, input1_quantized, scales[0],
        zero_points[0], input2_shape, input2_values, input2_quantized,
        scales[1], zero_points[1], test_shapes[i], golden, golden_quantized,
        scales[2], zero_points[2], kTfLiteActNone, output);
  }
}

TF_LITE_MICRO_TEST(QuantizedAddWithMixedBroadcastInt8) {
  const float scales[] = {0.1, 0.05, 0.1};
  const int zero_points[] = {-10, -5, 7};
  int8_t input1_quantized[tflite::testing::kBroadcastOutputDimsCount];
  int8_t input2_quantized[tflite::testing::kBroadcastOutputDimsCount];
  int8_t golden_quantized[tflite::testing::kBroadcastOutputDimsCount];
  int8_t output[tflite::testing::kBroadcastOutputDimsCount];

  for (int i = 0; i < tflite::testing::kBroadcastNumShapes; ++i) {
    tflite::testing::TestAddQuantized(
        tflite::testing::broadcast_input1_shape,
        tflite::testing::broadcast_input1_values, input1_quantized, scales[0],
        zero_points[0], tflite::testing::broadcast_input2_shapes[i],
        tflite::testing::broadcast_input2_values, input2_quantized, scales[1],
        zero_points[1], tflite::testing::broadcast_output_shapes[i],
        tflite::testing::broadcast_goldens[i], golden_quantized, scales[2],
        zero_points[2], kTfLiteActNone, output);
  }
}

TF_LITE_MICRO_TEST(QuantizedAddNoActivationInt16) {
  const float scales[] = {0.25, 0.5, 1.0};
  const int zero_points[] = {0, 0, 0};
  int inout_shape[] = {4, 1, 2, 2, 1};
  const float input1_values[] = {-2.01, -1.01, -0.01, 0.98};
  const float input2_values[] = {1.01, 2.01, 3.01, 4.02};
  const float golden_values[] = {-1, 1, 3, 5};

  constexpr int kOutputDimsCount = 4;
  int16_t input1_quantized[kOutputDimsCount];
  int16_t input2_quantized[kOutputDimsCount];
  int16_t golden_quantized[kOutputDimsCount];
  int16_t output[kOutputDimsCount];

  tflite::testing::TestAddQuantized(
      inout_shape, input1_values, input1_quantized, scales[0], zero_points[0],
      inout_shape, input2_values, input2_quantized, scales[1], zero_points[1],
      inout_shape, golden_values, golden_quantized, scales[2], zero_points[2],
      kTfLiteActNone, output);
}

TF_LITE_MICRO_TEST(QuantizedAddActivationRelu1Int16) {
  const float scales[] = {0.25, 0.5, 1.0};
  const int zero_points[] = {0, 0, 0};
  int inout_shape[] = {4, 1, 2, 2, 1};
  const float input1_values[] = {-2.01, -1.01, -0.01, 0.98};
  const float input2_values[] = {1.01, 1.99, 2.99, 4.02};
  const float golden_values[] = {-1, 1, 1, 1};

  constexpr int kOutputDimsCount = 4;
  int16_t input1_quantized[kOutputDimsCount];
  int16_t input2_quantized[kOutputDimsCount];
  int16_t golden_quantized[kOutputDimsCount];
  int16_t output[kOutputDimsCount];

  tflite::testing::TestAddQuantized(
      inout_shape, input1_values, input1_quantized, scales[0], zero_points[0],
      inout_shape, input2_values, input2_quantized, scales[1], zero_points[1],
      inout_shape, golden_values, golden_quantized, scales[2], zero_points[2],
      kTfLiteActReluN1To1, output);
}

TF_LITE_MICRO_TEST(QuantizedAddVariousInputShapesInt16) {
  const float scales[] = {0.1, 0.05, 0.1};
  const int zero_points[] = {0, 0, 0};

  constexpr int num_shapes = 4;
  constexpr int max_shape_size = 5;
  int test_shapes[num_shapes][max_shape_size] = {
      {1, 6},
      {2, 2, 3},
      {3, 2, 1, 3},
      {4, 1, 3, 1, 2},
  };

  const float input1_values[] = {-2.0, 0.2, 0.7, 0.8, 1.1, 2.0};
  const float input2_values[] = {0.1, 0.2, 0.3, 0.5, 1.1, 0.1};
  const float golden_values[] = {-1.9, 0.4, 1.0, 1.3, 2.2, 2.1};

  constexpr int kOutputDimsCount = 6;
  int16_t input1_quantized[kOutputDimsCount];
  int16_t input2_quantized[kOutputDimsCount];
  int16_t golden_quantized[kOutputDimsCount];
  int16_t output[kOutputDimsCount];

  for (int i = 0; i < num_shapes; i++) {
    tflite::testing::TestAddQuantized(
        test_shapes[i], input1_values, input1_quantized, scales[0],
        zero_points[0], test_shapes[i], input2_values, input2_quantized,
        scales[1], zero_points[1], test_shapes[i], golden_values,
        golden_quantized, scales[2], zero_points[2], kTfLiteActNone, output);
  }
}

TF_LITE_MICRO_TEST(QuantizedAddWithScalarBroadcastInt16) {
  const float input1_values[] = {-2.0, 0.2, 0.7, 0.8, 1.1, 2.0};
  int input2_shape[] = {0};
  const float input2_values[] = {0.1};
  const float golden[] = {-1.9, 0.3, 0.8, 0.9, 1.2, 2.1};

  constexpr int num_shapes = 4;
  constexpr int max_shape_size = 5;
  int test_shapes[num_shapes][max_shape_size] = {
      {1, 6},
      {2, 2, 3},
      {3, 2, 1, 3},
      {4, 1, 3, 1, 2},
  };

  const float scales[] = {0.1, 0.05, 0.05};
  const int zero_points[] = {0, 0, 0};

  constexpr int kOutputDimsCount = 6;
  int16_t input1_quantized[kOutputDimsCount];
  int16_t input2_quantized[kOutputDimsCount];
  int16_t golden_quantized[kOutputDimsCount];
  int16_t output[kOutputDimsCount];

  for (int i = 0; i < num_shapes; ++i) {
    tflite::testing::TestAddQuantized(
        test_shapes[i], input1_values, input1_quantized, scales[0],
        zero_points[0], input2_shape, input2_values, input2_quantized,
        scales[1], zero_points[1], test_shapes[i], golden, golden_quantized,
        scales[2], zero_points[2], kTfLiteActNone, output);
  }
}

TF_LITE_MICRO_TEST(QuantizedAddWithMixedBroadcastInt16) {
  const float scales[] = {0.1, 0.05, 0.1};
  const int zero_points[] = {0, 0, 0};
  int16_t input1_quantized[tflite::testing::kBroadcastOutputDimsCount];
  int16_t input2_quantized[tflite::testing::kBroadcastOutputDimsCount];
  int16_t golden_quantized[tflite::testing::kBroadcastOutputDimsCount];
  int16_t output[tflite::testing::kBroadcastOutputDimsCount];

  for (int i = 0; i < tflite::testing::kBroadcastNumShapes; ++i) {
    tflite::testing::TestAddQuantized(
        tflite::testing::broadcast_input1_shape,
        tflite::testing::broadcast_input1_values, input1_quantized, scales[0],
        zero_points[0], tflite::testing::broadcast_input2_shapes[i],
        tflite::testing::broadcast_input2_values, input2_quantized, scales[1],
        zero_points[1], tflite::testing::broadcast_output_shapes[i],
        tflite::testing::broadcast_goldens[i], golden_quantized, scales[2],
        zero_points[2], kTfLiteActNone, output);
  }
}

TF_LITE_MICRO_TEST(FloatAddBroadcast_2x2_1x2) {
  int input1_shape[] = {2, 2, 2};
  int input2_shape[] = {2, 1, 2};
  int output_shape[] = {2, 2, 2};
  const float input1_values[] = {1.0, 2.0, 3.0, 4.0};
  const float input2_values[] = {10.0, 20.0};
  const float golden_values[] = {11.0, 22.0, 13.0, 24.0};

  constexpr int kOutputDimsCount = 4;
  float output_data[kOutputDimsCount];
  tflite::testing::TestAddFloat(input1_shape, input1_values, input2_shape,
                                input2_values, output_shape, golden_values,
                                kTfLiteActNone, output_data);
}

TF_LITE_MICRO_TEST(FloatAddBroadcast_2x2_2x1) {
  int input1_shape[] = {2, 2, 2};
  int input2_shape[] = {2, 2, 1};
  int output_shape[] = {2, 2, 2};
  const float input1_values[] = {1.0, 2.0, 3.0, 4.0};
  const float input2_values[] = {10.0, 20.0};
  const float golden_values[] = {11.0, 12.0, 23.0, 24.0};

  constexpr int kOutputDimsCount = 4;
  float output_data[kOutputDimsCount];
  tflite::testing::TestAddFloat(input1_shape, input1_values, input2_shape,
                                input2_values, output_shape, golden_values,
                                kTfLiteActNone, output_data);
}

TF_LITE_MICRO_TEST(FloatAdd3D_Simple) {
  int inout_shape[] = {3, 2, 1, 2};
  const float input1_values[] = {1.0, 2.0, 3.0, 4.0};
  const float input2_values[] = {0.1, 0.2, 0.3, 0.4};
  const float golden_values[] = {1.1, 2.2, 3.3, 4.4};

  constexpr int kOutputDimsCount = 4;
  float output_data[kOutputDimsCount];
  tflite::testing::TestAddFloat(inout_shape, input1_values, inout_shape,
                                input2_values, inout_shape, golden_values,
                                kTfLiteActNone, output_data);
}

TF_LITE_MICRO_TEST(FloatAdd4D_Simple) {
  int inout_shape[] = {4, 1, 2, 1, 2};
  const float input1_values[] = {10.0, 20.0, 30.0, 40.0};
  const float input2_values[] = {1.0, 2.0, 3.0, 4.0};
  const float golden_values[] = {11.0, 22.0, 33.0, 44.0};

  constexpr int kOutputDimsCount = 4;
  float output_data[kOutputDimsCount];
  tflite::testing::TestAddFloat(inout_shape, input1_values, inout_shape,
                                input2_values, inout_shape, golden_values,
                                kTfLiteActNone, output_data);
}

TF_LITE_MICRO_TEST(FloatAddActivationReluN1To1_New) {
  int inout_shape[] = {2, 1, 2};
  const float input1_values[] = {-2.0, 5.0};
  const float input2_values[] = {0.5, 0.5};
  const float golden_values[] = {-1.0, 1.0};

  constexpr int kOutputDimsCount = 2;
  float output_data[kOutputDimsCount];
  tflite::testing::TestAddFloat(inout_shape, input1_values, inout_shape,
                                input2_values, inout_shape, golden_values,
                                kTfLiteActReluN1To1, output_data);
}

TF_LITE_MICRO_TEST(Int8AddSimple) {
  const float scales[] = {1.0, 1.0, 1.0};
  const int zero_points[] = {0, 0, 0};
  int inout_shape[] = {2, 1, 2};
  const float input1_values[] = {10, -10};
  const float input2_values[] = {5, -5};
  const float golden_values[] = {15, -15};

  constexpr int kOutputDimsCount = 2;
  int8_t input1_quantized[kOutputDimsCount];
  int8_t input2_quantized[kOutputDimsCount];
  int8_t golden_quantized[kOutputDimsCount];
  int8_t output[kOutputDimsCount];

  tflite::testing::TestAddQuantized(
      inout_shape, input1_values, input1_quantized, scales[0], zero_points[0],
      inout_shape, input2_values, input2_quantized, scales[1], zero_points[1],
      inout_shape, golden_values, golden_quantized, scales[2], zero_points[2],
      kTfLiteActNone, output);
}

TF_LITE_MICRO_TEST(Int8AddClampingPositive) {
  const float scales[] = {1.0, 1.0, 1.0};
  const int zero_points[] = {0, 0, 0};
  int inout_shape[] = {2, 1, 2};
  const float input1_values[] = {100, 100};
  const float input2_values[] = {100, 50};
  const float golden_values[] = {127, 127};

  constexpr int kOutputDimsCount = 2;
  int8_t input1_quantized[kOutputDimsCount];
  int8_t input2_quantized[kOutputDimsCount];
  int8_t golden_quantized[kOutputDimsCount];
  int8_t output[kOutputDimsCount];

  tflite::testing::TestAddQuantized(
      inout_shape, input1_values, input1_quantized, scales[0], zero_points[0],
      inout_shape, input2_values, input2_quantized, scales[1], zero_points[1],
      inout_shape, golden_values, golden_quantized, scales[2], zero_points[2],
      kTfLiteActNone, output);
}

TF_LITE_MICRO_TEST(Int8AddClampingNegative) {
  const float scales[] = {1.0, 1.0, 1.0};
  const int zero_points[] = {0, 0, 0};
  int inout_shape[] = {2, 1, 2};
  const float input1_values[] = {-100, -100};
  const float input2_values[] = {-50, -100};
  const float golden_values[] = {-128, -128};

  constexpr int kOutputDimsCount = 2;
  int8_t input1_quantized[kOutputDimsCount];
  int8_t input2_quantized[kOutputDimsCount];
  int8_t golden_quantized[kOutputDimsCount];
  int8_t output[kOutputDimsCount];

  tflite::testing::TestAddQuantized(
      inout_shape, input1_values, input1_quantized, scales[0], zero_points[0],
      inout_shape, input2_values, input2_quantized, scales[1], zero_points[1],
      inout_shape, golden_values, golden_quantized, scales[2], zero_points[2],
      kTfLiteActNone, output);
}

TF_LITE_MICRO_TEST(Int16AddSimple) {
  const float scales[] = {1.0, 1.0, 1.0};
  const int zero_points[] = {0, 0, 0};
  int inout_shape[] = {2, 1, 2};
  const float input1_values[] = {1000, -1000};
  const float input2_values[] = {500, -500};
  const float golden_values[] = {1500, -1500};

  constexpr int kOutputDimsCount = 2;
  int16_t input1_quantized[kOutputDimsCount];
  int16_t input2_quantized[kOutputDimsCount];
  int16_t golden_quantized[kOutputDimsCount];
  int16_t output[kOutputDimsCount];

  tflite::testing::TestAddQuantized(
      inout_shape, input1_values, input1_quantized, scales[0], zero_points[0],
      inout_shape, input2_values, input2_quantized, scales[1], zero_points[1],
      inout_shape, golden_values, golden_quantized, scales[2], zero_points[2],
      kTfLiteActNone, output);
}

TF_LITE_MICRO_TEST(Int16AddClamping) {
  const float scales[] = {1.0, 1.0, 1.0};
  const int zero_points[] = {0, 0, 0};
  int inout_shape[] = {2, 1, 2};
  const float input1_values[] = {30000, -30000};
  const float input2_values[] = {10000, -10000};
  const float golden_values[] = {32767, -32768};

  constexpr int kOutputDimsCount = 2;
  int16_t input1_quantized[kOutputDimsCount];
  int16_t input2_quantized[kOutputDimsCount];
  int16_t golden_quantized[kOutputDimsCount];
  int16_t output[kOutputDimsCount];

  tflite::testing::TestAddQuantized(
      inout_shape, input1_values, input1_quantized, scales[0], zero_points[0],
      inout_shape, input2_values, input2_quantized, scales[1], zero_points[1],
      inout_shape, golden_values, golden_quantized, scales[2], zero_points[2],
      kTfLiteActNone, output);
}

TF_LITE_MICRO_TESTS_END
