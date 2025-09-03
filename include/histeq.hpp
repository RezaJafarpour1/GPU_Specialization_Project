#pragma once
#include <string>
#include "image.hpp"

// Histogram Equalization for 8-bit grayscale.
// Returns empty string on success; error string on failure.
// If elapsed_ms != nullptr, it's filled with total GPU time (copies + kernels).
std::string histeq_cuda(const ImageU8 &in, ImageU8 &out, float *elapsed_ms = nullptr);
