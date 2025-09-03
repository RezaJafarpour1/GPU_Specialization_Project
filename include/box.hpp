#pragma once
#include <string>
#include "image.hpp"
#include "cuda_types.hpp"

// 3x3 mean blur (box) implemented with two-pass separable kernels.
// Returns empty string on success; error message on failure.
// If elapsed_ms is provided, we fill it with GPU time for both passes (copy+kernel+copy).
std::string box3_cuda(const ImageU8 &in, ImageU8 &out, float *elapsed_ms = nullptr);

// New: stream variant (re-uses an external tmp buffer: uint16_t*)
std::string box3_launch_stream(const uint8_t *d_in, uint8_t *d_out, uint16_t *d_tmp, int w, int h, cudaStream_t stream);
