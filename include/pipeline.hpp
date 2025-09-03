#pragma once
#include <vector>
#include <string>
#include <filesystem>
#include <fstream>
#include "image.hpp"

namespace fs = std::filesystem;

struct PipelineConfig
{
    int streams = 2;
    std::vector<std::string> ops; // order preserved
};

struct PipelineResult
{
    size_t processed = 0, skipped = 0, failed = 0;
    double total_wall_ms = 0.0;
};

std::string run_pipeline_streams(const std::vector<fs::path> &files,
                                 const fs::path &outdir,
                                 const PipelineConfig &pcfg,
                                 std::ofstream &log,
                                 std::ofstream &csv,
                                 PipelineResult &result);
