#pragma once
#include <vector>
#include <cstdint>

struct ImageU8
{
    int w = 0, h = 0;
    std::vector<uint8_t> data; // row-major, size = w*h
};
