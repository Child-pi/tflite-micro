/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_POW_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_POW_H_

#include <cmath>

#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/kernels/internal/types.h"

namespace tflite {
namespace reference_ops {

template <typename T>
inline void Pow(const RuntimeShape& input1_shape, const T* input1_data,
                const RuntimeShape& input2_shape, const T* input2_data,
                const RuntimeShape& output_shape, T* output_data) {
  const int flat_size =
      MatchingElementsSize(input1_shape, input2_shape, output_shape);
  for (int i = 0; i < flat_size; ++i) {
    output_data[i] = std::pow(input1_data[i], input2_data[i]);
  }
}

template <typename T>
inline void BroadcastPow4D(const RuntimeShape& input1_shape,
                           const T* input1_data,
                           const RuntimeShape& input2_shape,
                           const T* input2_data,
                           const RuntimeShape& output_shape, T* output_data) {
  const int dims1 = input1_shape.DimensionsCount();
  const int dims2 = input2_shape.DimensionsCount();
  const int dims_out = output_shape.DimensionsCount();

  // TFLite uses 4D broadcast in many places, or generic ND.
  // The original Pow code used BroadcastPow4D.
  // Let's implement a generic 4D broadcast or reuse generic broadcast if possible.
  // But standard BroadcastPow4D usually assumes 4D input.

  NdArrayDesc<4> desc1;
  NdArrayDesc<4> desc2;
  NdArrayDesc<4> output_desc;

  // Safe cast/pad to 4D
  int i = 0;
  for (; i < 4 - dims1; ++i) desc1.extents[i] = 1;
  for (int j = 0; j < dims1; ++j, ++i) desc1.extents[i] = input1_shape.Dims(j);

  i = 0;
  for (; i < 4 - dims2; ++i) desc2.extents[i] = 1;
  for (int j = 0; j < dims2; ++j, ++i) desc2.extents[i] = input2_shape.Dims(j);

  i = 0;
  for (; i < 4 - dims_out; ++i) output_desc.extents[i] = 1;
  for (int j = 0; j < dims_out; ++j, ++i) output_desc.extents[i] = output_shape.Dims(j);

  // Calculate strides
  desc1.strides[3] = 1;
  desc1.strides[2] = desc1.extents[3];
  desc1.strides[1] = desc1.extents[2] * desc1.strides[2];
  desc1.strides[0] = desc1.extents[1] * desc1.strides[1];

  desc2.strides[3] = 1;
  desc2.strides[2] = desc2.extents[3];
  desc2.strides[1] = desc2.extents[2] * desc2.strides[2];
  desc2.strides[0] = desc2.extents[1] * desc2.strides[1];

  output_desc.strides[3] = 1;
  output_desc.strides[2] = output_desc.extents[3];
  output_desc.strides[1] = output_desc.extents[2] * output_desc.strides[2];
  output_desc.strides[0] = output_desc.extents[1] * output_desc.strides[1];

  // Helper for 4D broadcast.
  // We can iterate over output and calculate indices for inputs.
  for (int b = 0; b < output_desc.extents[0]; ++b) {
    for (int y = 0; y < output_desc.extents[1]; ++y) {
      for (int x = 0; x < output_desc.extents[2]; ++x) {
        for (int c = 0; c < output_desc.extents[3]; ++c) {
           int in1_idx = 0;
           in1_idx += (b % desc1.extents[0]) * desc1.strides[0];
           in1_idx += (y % desc1.extents[1]) * desc1.strides[1];
           in1_idx += (x % desc1.extents[2]) * desc1.strides[2];
           in1_idx += (c % desc1.extents[3]) * desc1.strides[3];

           int in2_idx = 0;
           in2_idx += (b % desc2.extents[0]) * desc2.strides[0];
           in2_idx += (y % desc2.extents[1]) * desc2.strides[1];
           in2_idx += (x % desc2.extents[2]) * desc2.strides[2];
           in2_idx += (c % desc2.extents[3]) * desc2.strides[3];

           int out_idx = b * output_desc.strides[0] + y * output_desc.strides[1] + x * output_desc.strides[2] + c * output_desc.strides[3];

           output_data[out_idx] = std::pow(input1_data[in1_idx], input2_data[in2_idx]);
        }
      }
    }
  }
}

}  // namespace reference_ops
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_POW_H_
