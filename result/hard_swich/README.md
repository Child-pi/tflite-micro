# Hard Swish Int16 Verification

The unit test for `int16` HardSwish (`SimpleHardSwishTestInt16` in `hard_swish_test.cc`) uses a reference floating-point implementation to generate golden values.

## Golden Value Calculation

The HardSwish function is defined as:
$$f(x) = x \frac{\text{ReLU6}(x + 3)}{6} = x \frac{\min(\max(0, x+3), 6)}{6}$$

In the test:
1.  **Input Generation**: Uniform random floating-point values are generated within specified ranges (e.g., `[-10.0, 10.0]`).
2.  **Quantization**: These float inputs are quantized to `int16_t` using `SymmetricQuantize` (or similar logic via `ScaleFromMinMax`). The quantization parameters (scale, zero_point) are derived from the min/max range. For `int16`, we typically use symmetric quantization where `zero_point` is close to 0.
3.  **Reference Calculation**: The floating-point inputs are passed through the reference float HardSwish function:
    ```cpp
    result[i] = in * std::min(6.0f, std::max(0.0f, in + 3)) * (1.0f / 6.0f);
    ```
4.  **Golden Quantization**: The reference float outputs are quantized to `int16_t` using output quantization parameters.
5.  **Comparison**: The `int16` kernel output is dequantized back to float and compared against the reference float output (clamped to output range).

## Int16 Implementation Details

The `int16` implementation uses `int32_t` arithmetic to avoid overflow during intermediate calculations, which is critical because `int16` input values (difference from zero point) can span the full `int16` range and intermediate multiplication results exceed 16 bits.

Key steps in `HardSwishEvalInt16`:
1.  **Input scaling**: `input_value` (int32) is multiplied by `reluish_multiplier` to map `+/- 3.0` to `+/- 32768`.
2.  **Sigmoid approximation**: The result is saturated to `int16` range, then converted to `[0, 32767]` range representing `[0, 1]`.
3.  **Output scaling**: `input_value` is scaled to output domain using `output_multiplier`.
4.  **Final multiplication**: The scaled input is multiplied by the sigmoid factor.
5.  **Output shift**: `output_multiplier_exponent` is applied.

This ensures high precision and correctness across the dynamic range supported by `int16`.
