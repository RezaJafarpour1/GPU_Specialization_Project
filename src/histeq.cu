#include <cuda_runtime.h>
#include <cstdint>
#include <string>
#include <vector>
#include "histeq.hpp"
#include "cuda_utils.hpp"

namespace {
constexpr int TPB = 256; // threads per block

__global__ void kernel_histogram(const uint8_t* __restrict__ in,
                                 int n, unsigned int* __restrict__ g_hist)
{
    __shared__ unsigned int s_hist[256];
    // init shared hist
    for (int i = threadIdx.x; i < 256; i += blockDim.x) s_hist[i] = 0;
    __syncthreads();

    // grid-stride loop over pixels
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    while (idx < n) {
        atomicAdd(&s_hist[in[idx]], 1u);
        idx += stride;
    }
    __syncthreads();

    // reduce to global
    for (int i = threadIdx.x; i < 256; i += blockDim.x) {
        atomicAdd(&g_hist[i], s_hist[i]);
    }
}

__constant__ uint8_t d_lut[256];

__global__ void kernel_apply_lut(const uint8_t* __restrict__ in,
                                 uint8_t* __restrict__ out, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = d_lut[in[idx]];
}
} // namespace

std::string histeq_cuda(const ImageU8& in, ImageU8& out, float* elapsed_ms)
{
    if (in.w <= 0 || in.h <= 0 || (int)in.data.size() != in.w * in.h)
        return "invalid input image";

    out.w = in.w; out.h = in.h; out.data.resize(in.w * in.h);

    const int n = in.w * in.h;
    uint8_t *d_in = nullptr, *d_out = nullptr;
    unsigned int *d_hist = nullptr;

    CUDA_CHECK(cudaMalloc(&d_in,  n * sizeof(uint8_t)));
    CUDA_CHECK(cudaMalloc(&d_out, n * sizeof(uint8_t)));
    CUDA_CHECK(cudaMalloc(&d_hist, 256 * sizeof(unsigned int)));
    CUDA_CHECK(cudaMemset(d_hist, 0, 256 * sizeof(unsigned int)));
    CUDA_CHECK(cudaMemcpy(d_in, in.data.data(), n * sizeof(uint8_t), cudaMemcpyHostToDevice));

    cudaEvent_t evs, eve;
    if (elapsed_ms) { cudaEventCreate(&evs); cudaEventCreate(&eve); cudaEventRecord(evs); }

    // Launch histogram
    int blocks = (n + TPB - 1) / TPB;
    blocks = min(blocks, 1024); // cap blocks to keep atomics reasonable
    kernel_histogram<<<blocks, TPB>>>(d_in, n, d_hist);
    auto st = cudaGetLastError();
    if (st != cudaSuccess) {
        cudaFree(d_in); cudaFree(d_out); cudaFree(d_hist);
        return std::string("histogram launch failed: ") + cudaGetErrorString(st);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Bring histogram to host, compute CDF & LUT on host
    std::vector<unsigned int> h_hist(256);
    CUDA_CHECK(cudaMemcpy(h_hist.data(), d_hist, 256 * sizeof(unsigned int), cudaMemcpyDeviceToHost));

    // Compute CDF
    std::vector<unsigned int> cdf(256);
    unsigned int cum = 0; unsigned int cdf_min = 0;
    for (int i = 0; i < 256; ++i) {
        cum += h_hist[i];
        cdf[i] = cum;
        if (cdf_min == 0 && cum != 0) cdf_min = cum;
    }
    std::vector<uint8_t> h_lut(256);
    const unsigned int total = cdf.back();
    if (total == 0 || cdf_min == total) {
        // All pixels same or empty—identity LUT
        for (int i = 0; i < 256; ++i) h_lut[i] = static_cast<uint8_t>(i);
    } else {
        for (int i = 0; i < 256; ++i) {
            float num = float(cdf[i] - cdf_min);
            float den = float(total - cdf_min);
            int v = int(255.0f * (num / den) + 0.5f);
            if (v < 0) v = 0; if (v > 255) v = 255;
            h_lut[i] = static_cast<uint8_t>(v);
        }
    }

    // Upload LUT to constant memory and apply
    CUDA_CHECK(cudaMemcpyToSymbol(d_lut, h_lut.data(), 256 * sizeof(uint8_t)));
    kernel_apply_lut<<<(n + TPB - 1) / TPB, TPB>>>(d_in, d_out, n);
    st = cudaGetLastError();
    if (st != cudaSuccess) {
        cudaFree(d_in); cudaFree(d_out); cudaFree(d_hist);
        return std::string("apply LUT launch failed: ") + cudaGetErrorString(st);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    if (elapsed_ms) {
        cudaEventRecord(eve); cudaEventSynchronize(eve);
        cudaEventElapsedTime(elapsed_ms, evs, eve);
        cudaEventDestroy(evs); cudaEventDestroy(eve);
    }

    CUDA_CHECK(cudaMemcpy(out.data.data(), d_out, n * sizeof(uint8_t), cudaMemcpyDeviceToHost));
    cudaFree(d_in); cudaFree(d_out); cudaFree(d_hist);
    return {};
}
