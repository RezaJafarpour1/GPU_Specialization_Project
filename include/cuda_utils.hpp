#pragma once
#include <cuda_runtime.h>
#include <string>

inline std::string cuda_err_to_string(cudaError_t e)
{
    const char *s = cudaGetErrorString(e);
    return s ? std::string(s) : std::string("unknown cuda error");
}

#define CUDA_CHECK(call)                         \
    do                                           \
    {                                            \
        cudaError_t _e = (call);                 \
        if (_e != cudaSuccess)                   \
        {                                        \
            return std::string("CUDA error: ") + \
                   cuda_err_to_string(_e);       \
        }                                        \
    } while (0)
