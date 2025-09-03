#include <cuda_runtime.h>
#include <cstdint>
#include <string>
#include "box.hpp"
#include "cuda_utils.hpp"

namespace {
constexpr int BW = 16;
constexpr int BH = 16;

__device__ inline uint8_t clamp_read_u8(const uint8_t* src, int w, int h, int x, int y) {
    if (x < 0) x = 0; else if (x >= w) x = w - 1;
    if (y < 0) y = 0; else if (y >= h) y = h - 1;
    return src[y * w + x];
}
__device__ inline uint16_t clamp_read_u16(const uint16_t* src, int w, int h, int x, int y) {
    if (x < 0) x = 0; else if (x >= w) x = w - 1;
    if (y < 0) y = 0; else if (y >= h) y = h - 1;
    return src[y * w + x];
}

// Pass 1 (horizontal): writes the sum of 3 neighboring pixels (left+center+right) into tmp (16-bit).
__global__ void hpass_sum3(const uint8_t* __restrict__ in, uint16_t* __restrict__ tmp, int w, int h)
{
    __shared__ uint8_t tile[BH][BW + 2]; // halo of 1 on left/right
    const int tx = threadIdx.x, ty = threadIdx.y;
    const int x  = blockIdx.x * BW + tx;
    const int y  = blockIdx.y * BH + ty;

    // Center
    uint8_t c = (x < w && y < h) ? in[y * w + x] : 0;
    tile[ty][tx + 1] = c;

    // Halos
    if (tx == 0)       tile[ty][0]     = clamp_read_u8(in, w, h, x - 1, y);
    if (tx == BW - 1)  tile[ty][BW + 1]= clamp_read_u8(in, w, h, x + 1, y);

    __syncthreads();

    if (x >= w || y >= h) return;

    uint16_t sum3 = uint16_t(tile[ty][tx]) + uint16_t(tile[ty][tx + 1]) + uint16_t(tile[ty][tx + 2]);
    tmp[y * w + x] = sum3;
}

// Pass 2 (vertical): takes tmp sum3 values, makes a 3-high sum, divides by 9, writes 8-bit output.
__global__ void vpass_div9(const uint16_t* __restrict__ tmp, uint8_t* __restrict__ out, int w, int h)
{
    __shared__ uint16_t tile[BH + 2][BW]; // halo of 1 on top/bottom
    const int tx = threadIdx.x, ty = threadIdx.y;
    const int x  = blockIdx.x * BW + tx;
    const int y  = blockIdx.y * BH + ty;

    // Center
    uint16_t c = (x < w && y < h) ? tmp[y * w + x] : 0;
    tile[ty + 1][tx] = c;

    // Halos
    if (ty == 0)       tile[0][tx]        = clamp_read_u16(tmp, w, h, x, y - 1);
    if (ty == BH - 1)  tile[BH + 1][tx]   = clamp_read_u16(tmp, w, h, x, y + 1);

    __syncthreads();

    if (x >= w || y >= h) return;

    uint32_t sum9 = uint32_t(tile[ty][tx]) + uint32_t(tile[ty + 1][tx]) + uint32_t(tile[ty + 2][tx]);
    // sum9 currently = (left+center+right) for rows y-1,y,y+1 => true 3x3 box sum
    uint32_t v = sum9 / 9u;
    if (v > 255u) v = 255u;
    out[y * w + x] = static_cast<uint8_t>(v);
}
} // namespace

std::string box3_cuda(const ImageU8& in, ImageU8& out, float* elapsed_ms)
{
    if (in.w <= 0 || in.h <= 0 || (int)in.data.size() != in.w * in.h)
        return "invalid input image";

    out.w = in.w; out.h = in.h; out.data.resize(in.w * in.h);

    uint8_t  *d_in  = nullptr, *d_out = nullptr;
    uint16_t *d_tmp = nullptr;
    size_t npx   = static_cast<size_t>(in.w) * in.h;
    size_t bytes_u8  = npx * sizeof(uint8_t);
    size_t bytes_u16 = npx * sizeof(uint16_t);

    CUDA_CHECK(cudaMalloc(&d_in,  bytes_u8));
    CUDA_CHECK(cudaMalloc(&d_out, bytes_u8));
    CUDA_CHECK(cudaMalloc(&d_tmp, bytes_u16));
    CUDA_CHECK(cudaMemcpy(d_in, in.data.data(), bytes_u8, cudaMemcpyHostToDevice));

    dim3 block(BW, BH);
    dim3 grid((in.w + BW - 1) / BW, (in.h + BH - 1) / BH);

    cudaEvent_t evs, eve;
    if (elapsed_ms) { cudaEventCreate(&evs); cudaEventCreate(&eve); cudaEventRecord(evs); }

    hpass_sum3<<<grid, block>>>(d_in, d_tmp, in.w, in.h);
    auto st = cudaGetLastError();
    if (st != cudaSuccess) {
        cudaFree(d_in); cudaFree(d_out); cudaFree(d_tmp);
        return std::string("box hpass launch failed: ") + cudaGetErrorString(st);
    }

    vpass_div9<<<grid, block>>>(d_tmp, d_out, in.w, in.h);
    st = cudaGetLastError();
    if (st != cudaSuccess) {
        cudaFree(d_in); cudaFree(d_out); cudaFree(d_tmp);
        return std::string("box vpass launch failed: ") + cudaGetErrorString(st);
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    if (elapsed_ms) {
        cudaEventRecord(eve); cudaEventSynchronize(eve);
        cudaEventElapsedTime(elapsed_ms, evs, eve);
        cudaEventDestroy(evs); cudaEventDestroy(eve);
    }

    CUDA_CHECK(cudaMemcpy(out.data.data(), d_out, bytes_u8, cudaMemcpyDeviceToHost));
    cudaFree(d_in); cudaFree(d_out); cudaFree(d_tmp);
    return {};
}


//extern __global__ void hpass_sum3(const uint8_t* __restrict__, uint16_t* __restrict__, int, int);
//extern __global__ void vpass_div9(const uint16_t* __restrict__, uint8_t* __restrict__, int, int);

std::string box3_launch_stream(const uint8_t* d_in, uint8_t* d_out, uint16_t* d_tmp, int w, int h, cudaStream_t stream)
{
    if (!d_in || !d_out || !d_tmp || w <= 0 || h <= 0) return "box3_launch_stream: bad args";
    constexpr int BW = 16, BH = 16;
    dim3 block(BW, BH);
    dim3 grid((w + BW - 1) / BW, (h + BH - 1) / BH);
    hpass_sum3<<<grid, block, 0, stream>>>(d_in, d_tmp, w, h);
    auto st = cudaGetLastError();
    if (st != cudaSuccess) return std::string("box hpass launch failed: ") + cudaGetErrorString(st);
    vpass_div9<<<grid, block, 0, stream>>>(d_tmp, d_out, w, h);
    st = cudaGetLastError();
    if (st != cudaSuccess) return std::string("box vpass launch failed: ") + cudaGetErrorString(st);
    return {};
}
