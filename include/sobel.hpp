#pragma once
#include <string>
#include "image.hpp"

// Return empty string on success, or error message on failure.
std::string sobel_cuda(const ImageU8 &in, ImageU8 &out);
