#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <unordered_set>
#include <filesystem>
#include <optional>
#include <chrono>
#include <ctime>
#include <cctype>
#include <cstdlib>
#include <sstream>
#include <iomanip>

#include "image.hpp"
#include "pgm.hpp"
#include "sobel.hpp"
#include "box.hpp"

namespace fs = std::filesystem;

struct Config
{
    fs::path input_dir;
    fs::path output_dir;
    std::vector<std::string> ops; // "sobel","box","gauss","histeq","invert"
    int batch = 256;
    bool recursive = false;
    bool dry_run = false;
};

static void print_usage(const char *prog)
{
    std::cout
        << "Usage: " << prog << " --input_dir <dir> --output_dir <dir> "
        << "[--ops sobel,box,gauss,histeq,invert] [--batch N] [--recursive] [--dry_run] [--help]\n\n"
        << "Options:\n"
        << "  --input_dir <dir>    Directory of input images\n"
        << "  --output_dir <dir>   Directory to write outputs (created if missing)\n"
        << "  --ops <list>         Comma-separated ops. Implemented: sobel, invert\n"
        << "  --batch N            Number of files to process per wave (default 256)\n"
        << "  --recursive          Recurse into subdirectories\n"
        << "  --dry_run            Parse, validate, and enumerate—without processing\n"
        << "  --help               Show this help\n";
}

static std::vector<std::string> split_csv(const std::string &s)
{
    std::vector<std::string> out;
    std::string cur;
    for (char c : s)
    {
        if (c == ',')
        {
            if (!cur.empty())
                out.push_back(cur);
            cur.clear();
        }
        else
            cur.push_back(c);
    }
    if (!cur.empty())
        out.push_back(cur);
    return out;
}

static std::optional<Config> parse_args(int argc, char **argv)
{
    Config cfg;
    for (int i = 1; i < argc; ++i)
    {
        std::string a = argv[i];
        auto need_next = [&](const char *flag) -> std::string {
            if (i + 1 >= argc)
            {
                std::cerr << "Missing value for " << flag << "\n";
                return {};
            }
            return argv[++i];
        };

        if (a == "--help" || a == "-h")
        {
            print_usage(argv[0]);
            std::exit(0);
        }
        else if (a == "--input_dir")
        {
            cfg.input_dir = need_next("--input_dir");
        }
        else if (a == "--output_dir")
        {
            cfg.output_dir = need_next("--output_dir");
        }
        else if (a == "--ops")
        {
            cfg.ops = split_csv(need_next("--ops"));
        }
        else if (a == "--batch")
        {
            cfg.batch = std::stoi(need_next("--batch"));
        }
        else if (a == "--recursive")
        {
            cfg.recursive = true;
        }
        else if (a == "--dry_run")
        {
            cfg.dry_run = true;
        }
        else
        {
            std::cerr << "Unknown flag: " << a << "\n";
            return std::nullopt;
        }
    }

    if (cfg.input_dir.empty() || cfg.output_dir.empty())
    {
        std::cerr << "Error: --input_dir and --output_dir are required.\n";
        return std::nullopt;
    }
    return cfg;
}

static bool is_image_file(const fs::path &p)
{
    static const std::unordered_set<std::string> exts = {
        ".pgm", ".ppm", ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"};
    std::string ext = p.extension().string();
    for (char &ch : ext)
    {
        ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    }
    return exts.count(ext) > 0;
}

// CPU placeholder op
static void cpu_invert(ImageU8 &img)
{
    for (auto &px : img.data)
        px = static_cast<uint8_t>(255 - px);
}

int main(int argc, char **argv)
{
    auto cfgOpt = parse_args(argc, argv);
    if (!cfgOpt)
    {
        print_usage(argv[0]);
        return 2;
    }
    Config cfg = *cfgOpt;

    if (!fs::exists(cfg.input_dir) || !fs::is_directory(cfg.input_dir))
    {
        std::cerr << "Error: input_dir does not exist or is not a directory: "
                  << cfg.input_dir << "\n";
        return 2;
    }
    std::error_code ec;
    fs::create_directories(cfg.output_dir, ec);
    if (ec)
    {
        std::cerr << "Error: cannot create output_dir: " << cfg.output_dir
                  << " (" << ec.message() << ")\n";
        return 2;
    }

    const std::unordered_set<std::string> allowed = {"sobel", "box", "gauss", "histeq", "invert"};
    for (const auto &op : cfg.ops)
    {
        if (!allowed.count(op))
        {
            std::cerr << "Warning: unknown op '" << op << "'\n";
        }
    }

    std::vector<fs::path> files;
    if (cfg.recursive)
    {
        for (auto &e : fs::recursive_directory_iterator(cfg.input_dir))
            if (e.is_regular_file() && is_image_file(e.path()))
                files.push_back(e.path());
    }
    else
    {
        for (auto &e : fs::directory_iterator(cfg.input_dir))
            if (e.is_regular_file() && is_image_file(e.path()))
                files.push_back(e.path());
    }

    auto now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    fs::path log_path = cfg.output_dir / "log.txt";
    std::ofstream log(log_path);
    log << "gpu_pipeline session log\n";
    log << "timestamp: " << std::ctime(&now);
    log << "input_dir: " << cfg.input_dir << "\n";
    log << "output_dir: " << cfg.output_dir << "\n";
    log << "ops: ";
    for (size_t i = 0; i < cfg.ops.size(); ++i)
    {
        log << cfg.ops[i] << (i + 1 < cfg.ops.size() ? "," : "");
    }
    log << "\n";

    std::cout << "[INFO] Found " << files.size() << " image(s).\n";

    if (cfg.dry_run)
    {
        log << "files_found: " << files.size() << "\n";
        std::cout << "[OK] Dry run only. Log: " << log_path << "\n";
        return 0;
    }

    size_t processed = 0, skipped = 0, failed = 0;

    // CSV metrics (per image/op)
    fs::path csv_path = cfg.output_dir / "metrics.csv";
    std::ofstream csv(csv_path);
    csv << "filename,op,ms\n";

    for (const auto &in_path : files)
    {
        if (in_path.extension() != ".pgm" && in_path.extension() != ".PGM")
        {
            ++skipped;
            log << "SKIP (currently only PGM handled): " << in_path.filename().string() << "\n";
            continue;
        }

        ImageU8 img;
        std::string err;
        if (!read_pgm(in_path, img, err))
        {
            ++failed;
            log << "FAIL read " << in_path.filename().string() << " : " << err << "\n";
            continue;
        }

        // Apply ops in the given order with simple wall timing around each call
        bool any_error = false;
        for (const auto &op : cfg.ops)
        {
            using clock = std::chrono::steady_clock;
            auto t0 = clock::now();
            float gpu_ms = 0.0f;

            if (op == "invert")
            {
                cpu_invert(img);
            }
            else if (op == "sobel")
            {
                ImageU8 out;
                std::string e = sobel_cuda(img, out);
                if (!e.empty())
                {
                    any_error = true;
                    log << "FAIL sobel " << in_path.filename().string() << " : " << e << "\n";
                    break;
                }
                img = std::move(out);
            }
            else if (op == "box")
            {
                ImageU8 out;
                std::string e = box3_cuda(img, out, &gpu_ms);
                if (!e.empty())
                {
                    any_error = true;
                    log << "FAIL box " << in_path.filename().string() << " : " << e << "\n";
                    break;
                }
                img = std::move(out);
            }
            else if (op == "gauss" || op == "histeq")
            {
                log << "TODO (not implemented yet): " << op << " for " << in_path.filename().string() << "\n";
            }
            else
            {
                log << "Warning: unknown op '" << op << "'\n";
            }

            auto t1 = clock::now();
            double wall_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

            // prefer CUDA event time if we got it; otherwise wall-clock
            double used_ms = (gpu_ms > 0.0f) ? gpu_ms : wall_ms;
            csv << in_path.filename().string() << "," << op << "," << std::fixed << std::setprecision(3) << used_ms << "\n";
        }

        // Write output even if some ops were TODO (we still save the last successful image)
        fs::path out_path = cfg.output_dir / in_path.filename();
        if (!write_pgm(out_path, img, err, /*binary*/ true))
        {
            ++failed;
            log << "FAIL write " << out_path.filename().string() << " : " << err << "\n";
            continue;
        }
        if (!any_error)
            ++processed;
        log << "OK   " << in_path.filename().string() << " -> " << out_path.filename().string() << "\n";
    }

    log << "summary: processed=" << processed
        << " skipped=" << skipped
        << " failed=" << failed << "\n";

    std::cout << "[DONE] processed=" << processed
              << " skipped=" << skipped
              << " failed=" << failed
              << "  Log: " << log_path << "\n"
              << "[INFO] Metrics CSV: " << csv_path << "\n";
    return (failed == 0) ? 0 : 1;
}
