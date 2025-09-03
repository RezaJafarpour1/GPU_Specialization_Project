#pragma once
#include <string>
#include "image.hpp"
#include "cuda_types.hpp"

// Return empty string on success, or error message on failure.
std::string sobel_cuda(const ImageU8 &in, ImageU8 &out);

// New: launch on an existing stream using pre-allocated device buffers.
std::string sobel_launch_stream(const uint8_t *d_in, uint8_t *d_out, int w, int h, cudaStream_t stream);