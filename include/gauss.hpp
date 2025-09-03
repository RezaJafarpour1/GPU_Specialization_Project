#pragma once
#include <string>
#include "image.hpp"
#include "cuda_types.hpp"

// 5x5 Gaussian (weights 1 4 6 4 1, separable).
// Returns empty string on success; error string on failure.
// If elapsed_ms != nullptr, it's filled with total GPU time (copies + kernels).
std::string gauss5_cuda(const ImageU8 &in, ImageU8 &out, float *elapsed_ms = nullptr);

// New: stream variant (uses external tmp uint16_t buffer)
std::string gauss5_launch_stream(const uint8_t *d_in, uint8_t *d_out, uint16_t *d_tmp, int w, int h, cudaStream_t stream);