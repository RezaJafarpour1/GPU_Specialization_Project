#pragma once
#include <string>
#include "image.hpp"

// 3x3 mean blur (box) implemented with two-pass separable kernels.
// Returns empty string on success; error message on failure.
// If elapsed_ms is provided, we fill it with GPU time for both passes (copy+kernel+copy).
std::string box3_cuda(const ImageU8 &in, ImageU8 &out, float *elapsed_ms = nullptr);
