#pragma once
#include <string>
#include "image.hpp"
#include "cuda_types.hpp"

// Histogram Equalization for 8-bit grayscale.
// Returns empty string on success; error string on failure.
// If elapsed_ms != nullptr, it's filled with total GPU time (copies + kernels).
std::string histeq_cuda(const ImageU8 &in, ImageU8 &out, float *elapsed_ms = nullptr);

// stream variant (uses external global histogram buffer; LUT build happens on host between kernels)
std::string histeq_launch_stream(uint8_t *d_in, uint8_t *d_out, unsigned int *d_hist, int w, int h, cudaStream_t stream, std::string *err_out = nullptr);
