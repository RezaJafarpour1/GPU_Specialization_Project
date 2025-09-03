#include <cuda_runtime.h>
#include <cstdint>
#include <string>

#include "sobel.hpp"
#include "cuda_utils.hpp"

namespace {
constexpr int BLOCK_W = 16;
constexpr int BLOCK_H = 16;

__constant__ int8_t KX[9] = { -1, 0, 1,
                              -2, 0, 2,
                              -1, 0, 1 };
__constant__ int8_t KY[9] = { -1,-2,-1,
                               0, 0, 0,
                               1, 2, 1 };

__device__ inline uint8_t clamp_read(const uint8_t* src, int w, int h, int x, int y) {
    if (x < 0) x = 0; else if (x >= w) x = w - 1;
    if (y < 0) y = 0; else if (y >= h) y = h - 1;
    return src[y * w + x];
}

__global__ void sobel_kernel(const uint8_t* __restrict__ in,
                             uint8_t* __restrict__ out,
                             int w, int h)
{
    __shared__ uint8_t tile[BLOCK_H + 2][BLOCK_W + 2]; // 1-pixel halo

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int x  = blockIdx.x * BLOCK_W + tx;
    const int y  = blockIdx.y * BLOCK_H + ty;

    tile[ty + 1][tx + 1] = (x < w && y < h) ? in[y * w + x] : 0;

    if (tx == 0)                  tile[ty + 1][0]            = clamp_read(in, w, h, x - 1, y);
    if (tx == BLOCK_W - 1)        tile[ty + 1][BLOCK_W + 1]  = clamp_read(in, w, h, x + 1, y);
    if (ty == 0)                  tile[0][tx + 1]            = clamp_read(in, w, h, x, y - 1);
    if (ty == BLOCK_H - 1)        tile[BLOCK_H + 1][tx + 1]  = clamp_read(in, w, h, x, y + 1);

    if (tx == 0 && ty == 0)                               tile[0][0]                         = clamp_read(in, w, h, x - 1, y - 1);
    if (tx == BLOCK_W - 1 && ty == 0)                     tile[0][BLOCK_W + 1]               = clamp_read(in, w, h, x + 1, y - 1);
    if (tx == 0 && ty == BLOCK_H - 1)                     tile[BLOCK_H + 1][0]               = clamp_read(in, w, h, x - 1, y + 1);
    if (tx == BLOCK_W - 1 && ty == BLOCK_H - 1)           tile[BLOCK_H + 1][BLOCK_W + 1]     = clamp_read(in, w, h, x + 1, y + 1);

    __syncthreads();

    if (x >= w || y >= h) return;

    if (x == 0 || y == 0 || x == w - 1 || y == h - 1) {
        out[y * w + x] = 0;
        return;
    }

    int gx = 0, gy = 0;
    #pragma unroll
    for (int ky = -1; ky <= 1; ++ky) {
        #pragma unroll
        for (int kx = -1; kx <= 1; ++kx) {
            int p = tile[ty + 1 + ky][tx + 1 + kx];
            int kidx = (ky + 1) * 3 + (kx + 1);
            gx += p * KX[kidx];
            gy += p * KY[kidx];
        }
    }
    int mag = abs(gx) + abs(gy);
    if (mag > 255) mag = 255;
    out[y * w + x] = static_cast<uint8_t>(mag);
}
} // namespace

std::string sobel_cuda(const ImageU8& in, ImageU8& out)
{
    if (in.w <= 0 || in.h <= 0 || (int)in.data.size() != in.w * in.h)
        return "invalid input image";

    out.w = in.w; out.h = in.h; out.data.resize(in.w * in.h);

    uint8_t *d_in = nullptr, *d_out = nullptr;
    size_t bytes = static_cast<size_t>(in.w) * in.h;

    CUDA_CHECK(cudaMalloc(&d_in,  bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    CUDA_CHECK(cudaMemcpy(d_in, in.data.data(), bytes, cudaMemcpyHostToDevice));

    dim3 block(BLOCK_W, BLOCK_H);
    dim3 grid((in.w + BLOCK_W - 1) / BLOCK_W,
              (in.h + BLOCK_H - 1) / BLOCK_H);

    sobel_kernel<<<grid, block>>>(d_in, d_out, in.w, in.h);
    auto st = cudaGetLastError();
    if (st != cudaSuccess) {
        cudaFree(d_in); cudaFree(d_out);
        return std::string("launch failed: ") + cudaGetErrorString(st);
    }
    st = cudaDeviceSynchronize();
    if (st != cudaSuccess) {
        cudaFree(d_in); cudaFree(d_out);
        return std::string("kernel error: ") + cudaGetErrorString(st);
    }

    CUDA_CHECK(cudaMemcpy(out.data.data(), d_out, bytes, cudaMemcpyDeviceToHost));

    cudaFree(d_in); cudaFree(d_out);
    return {};
}


std::string sobel_launch_stream(const uint8_t* d_in, uint8_t* d_out, int w, int h, cudaStream_t stream)
{
    if (!d_in || !d_out || w <= 0 || h <= 0) return "sobel_launch_stream: bad args";
    constexpr int BLOCK_W = 16, BLOCK_H = 16;
    dim3 block(BLOCK_W, BLOCK_H);
    dim3 grid((w + BLOCK_W - 1) / BLOCK_W, (h + BLOCK_H - 1) / BLOCK_H);
    sobel_kernel<<<grid, block, 0, stream>>>(d_in, d_out, w, h);
    auto st = cudaGetLastError();
    if (st != cudaSuccess) return std::string("sobel launch failed: ") + cudaGetErrorString(st);
    return {};
}
