#include <cuda_runtime.h>
#include <vector>
#include <string>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <utility>
#include <cstdint>
#include <algorithm>

#include "pipeline.hpp"
#include "pgm.hpp"
#include "sobel.hpp"
#include "box.hpp"
#include "gauss.hpp"
#include "histeq.hpp"
#include "cuda_utils.hpp"

#include <filesystem>
namespace fs = std::filesystem; 

struct StreamCtx {
    cudaStream_t s = nullptr;
    uint8_t  *d_a = nullptr, *d_b = nullptr; // ping-pong
    uint16_t *d_tmp = nullptr;               // for box/gauss
    unsigned int *d_hist = nullptr;          // for histeq
    size_t cap_pixels = 0;

    // Pending job state
    bool busy = false;
    fs::path in_path, out_path;
    ImageU8 h_in, h_out;
    std::vector<std::pair<std::string, std::pair<cudaEvent_t, cudaEvent_t>>> evs; // per-op timing
};

static void free_ctx(StreamCtx& c) {
    if (c.s) cudaStreamSynchronize(c.s);
    if (c.d_a) cudaFree(c.d_a);
    if (c.d_b) cudaFree(c.d_b);
    if (c.d_tmp) cudaFree(c.d_tmp);
    if (c.d_hist) cudaFree(c.d_hist);
    if (c.s) cudaStreamDestroy(c.s);
    c = {};
}

static std::string ensure_capacity(StreamCtx& c, size_t pixels) {
    if (pixels <= c.cap_pixels) return {};
    if (c.d_a) { cudaFree(c.d_a); c.d_a = nullptr; }
    if (c.d_b) { cudaFree(c.d_b); c.d_b = nullptr; }
    if (c.d_tmp) { cudaFree(c.d_tmp); c.d_tmp = nullptr; }
    if (c.d_hist) { cudaFree(c.d_hist); c.d_hist = nullptr; }

    CUDA_CHECK(cudaMalloc(&c.d_a, pixels * sizeof(uint8_t)));
    CUDA_CHECK(cudaMalloc(&c.d_b, pixels * sizeof(uint8_t)));
    CUDA_CHECK(cudaMalloc(&c.d_tmp, pixels * sizeof(uint16_t)));
    CUDA_CHECK(cudaMalloc(&c.d_hist, 256 * sizeof(unsigned int)));
    c.cap_pixels = pixels;
    return {};
}

// Flush a completed stream: wait, collect timings, write file, log+csv.
static void flush_stream(StreamCtx& c, std::ofstream& log, std::ofstream& csv, PipelineResult& res) {
    if (!c.busy) return;
    cudaStreamSynchronize(c.s);

    // collect timings
    for (auto& kv : c.evs) {
        float ms = 0.f;
        cudaEventElapsedTime(&ms, kv.second.first, kv.second.second);
        csv << c.in_path.filename().string() << "," << kv.first << "," << ms << "\n";
        cudaEventDestroy(kv.second.first);
        cudaEventDestroy(kv.second.second);
    }
    c.evs.clear();

    // write PGM
    std::string err;
    if (!write_pgm(c.out_path, c.h_out, err, true)) {
        ++res.failed;
        log << "FAIL write " << c.out_path.filename().string() << " : " << err << "\n";
    } else {
        ++res.processed;
        log << "OK   " << c.in_path.filename().string() << " -> " << c.out_path.filename().string() << "\n";
    }
    c.busy = false;
}

std::string run_pipeline_streams(const std::vector<fs::path>& files,
                                 const fs::path& outdir,
                                 const PipelineConfig& pcfg,
                                 std::ofstream& log,
                                 std::ofstream& csv,
                                 PipelineResult& result)
{
    result = {};
    if (pcfg.streams <= 0) return "streams must be >= 1";
    std::vector<StreamCtx> S(pcfg.streams);

    // create streams
    for (auto& c : S) {
        if (cudaStreamCreate(&c.s) != cudaSuccess) return "failed to create stream";
    }

    auto t0 = std::chrono::steady_clock::now();

    for (size_t i = 0; i < files.size(); ++i) {
        const fs::path in_path = files[i];
        if (!(in_path.extension() == ".pgm" || in_path.extension() == ".PGM")) {
            ++result.skipped;
            log << "SKIP (streams mode supports PGM) : " << in_path.filename().string() << "\n";
            continue;
        }

        // choose a stream in round-robin; if busy, flush it first
        StreamCtx& c = S[i % S.size()];
        if (c.busy) flush_stream(c, log, csv, result);

        // read input
        c.in_path = in_path;
        c.out_path = outdir / in_path.filename();
        std::string err;
        if (!read_pgm(in_path, c.h_in, err)) {
            ++result.failed;
            log << "FAIL read " << in_path.filename().string() << " : " << err << "\n";
            c.busy = false;
            continue;
        }
        c.h_out.w = c.h_in.w; c.h_out.h = c.h_in.h; c.h_out.data.resize(c.h_in.w * c.h_in.h);

        const int w = c.h_in.w, h = c.h_in.h;
        const size_t pixels = static_cast<size_t>(w) * h;

        if (auto e = ensure_capacity(c, pixels); !e.empty()) {
            ++result.failed; log << "FAIL alloc " << in_path.filename().string() << " : " << e << "\n";
            c.busy = false; continue;
        }

        // Optional CPU invert: perform before H2D to keep kernels simple
        ImageU8* src_host = &c.h_in;

        // H2D
        CUDA_CHECK(cudaMemcpyAsync(c.d_a, src_host->data.data(), pixels * sizeof(uint8_t),
                                cudaMemcpyHostToDevice, c.s));
        uint8_t* d_cur_in  = c.d_a;
        uint8_t* d_cur_out = c.d_b;

        // per-op launches + per-op timing
        c.evs.clear();
        for (const auto& op : pcfg.ops) {
            // Set up events for timing this op
            cudaEvent_t es, ee; cudaEventCreate(&es); cudaEventCreate(&ee);
            cudaEventRecord(es, c.s);

            std::string e;
            if (op == "sobel") {
                e = sobel_launch_stream(d_cur_in, d_cur_out, w, h, c.s);
                std::swap(d_cur_in, d_cur_out);
            } else if (op == "box") {
                e = box3_launch_stream(d_cur_in, d_cur_out, c.d_tmp, w, h, c.s);
                std::swap(d_cur_in, d_cur_out);
            } else if (op == "gauss") {
                e = gauss5_launch_stream(d_cur_in, d_cur_out, c.d_tmp, w, h, c.s);
                std::swap(d_cur_in, d_cur_out);
            } else if (op == "histeq") {
                e = histeq_launch_stream(d_cur_in, d_cur_out, c.d_hist, w, h, c.s);
                std::swap(d_cur_in, d_cur_out);
            } else if (op == "invert") {
                // tiny device invert kernel not added — do it on host beforehand if needed.
                // For streams mode, treat as no-op here.
            } else {
                log << "TODO (streams): " << op << " for " << in_path.filename().string() << "\n";
            }

            if (!e.empty()) {
                // Record failure for this image; still try to get it back and write what we have
                log << "FAIL " << op << " " << in_path.filename().string() << " : " << e << "\n";
            }

            cudaEventRecord(ee, c.s);
            c.evs.emplace_back(op, std::make_pair(es, ee));
        }

        // D2H of final buffer
        CUDA_CHECK(cudaMemcpyAsync(c.h_out.data.data(), d_cur_in, pixels * sizeof(uint8_t),
                           cudaMemcpyDeviceToHost, c.s));

        // mark busy; will flush on next reuse or at the end
        c.busy = true;
    }

    // Flush remaining busy streams
    for (auto& c : S) if (c.busy) flush_stream(c, log, csv, result);

    auto t1 = std::chrono::steady_clock::now();
    result.total_wall_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    for (auto& c : S) free_ctx(c);
    return {};
}
