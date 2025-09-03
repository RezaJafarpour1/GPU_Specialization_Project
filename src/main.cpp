#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <unordered_set>
#include <filesystem>
#include <optional>
#include <chrono>
#include <ctime>   // std::ctime
#include <cctype>  // std::tolower
#include <cstdlib> // std::exit

namespace fs = std::filesystem;

struct Config
{
    fs::path input_dir;
    fs::path output_dir;
    std::vector<std::string> ops; // e.g., {"sobel","box","histeq"}
    int batch = 256;
    bool recursive = false;
    bool dry_run = false;
};

static void print_usage(const char *prog)
{
    std::cout
        << "Usage: " << prog << " --input_dir <dir> --output_dir <dir> "
        << "[--ops sobel,box,histeq] [--batch N] [--recursive] [--dry_run] [--help]\n\n"
        << "Options:\n"
        << "  --input_dir <dir>    Directory of input images\n"
        << "  --output_dir <dir>   Directory to write outputs (created if missing)\n"
        << "  --ops <list>         Comma-separated ops; e.g., sobel,box,histeq (validated only)\n"
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

int main(int argc, char **argv)
{
    auto cfgOpt = parse_args(argc, argv);
    if (!cfgOpt)
    {
        print_usage(argv[0]);
        return 2;
    }
    Config cfg = *cfgOpt;

    // Validate input_dir
    if (!fs::exists(cfg.input_dir) || !fs::is_directory(cfg.input_dir))
    {
        std::cerr << "Error: input_dir does not exist or is not a directory: "
                  << cfg.input_dir << "\n";
        return 2;
    }
    // Prepare output_dir
    std::error_code ec;
    fs::create_directories(cfg.output_dir, ec);
    if (ec)
    {
        std::cerr << "Error: cannot create output_dir: " << cfg.output_dir
                  << " (" << ec.message() << ")\n";
        return 2;
    }

    // Validate ops (no actual processing yet)
    const std::unordered_set<std::string> allowed = {"sobel", "box", "gauss", "histeq"};
    for (const auto &op : cfg.ops)
    {
        if (!allowed.count(op))
        {
            std::cerr << "Warning: unknown op '" << op << "' (allowed: sobel, box, gauss, histeq)\n";
        }
    }

    // Enumerate images
    std::vector<fs::path> files;
    if (cfg.recursive)
    {
        for (auto &e : fs::recursive_directory_iterator(cfg.input_dir))
        {
            if (e.is_regular_file() && is_image_file(e.path()))
                files.push_back(e.path());
        }
    }
    else
    {
        for (auto &e : fs::directory_iterator(cfg.input_dir))
        {
            if (e.is_regular_file() && is_image_file(e.path()))
                files.push_back(e.path());
        }
    }

    // Minimal “proof” scaffolding: write a log stub
    auto now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    fs::path log_path = cfg.output_dir / "log.txt";
    std::ofstream log(log_path);
    log << "gpu_pipeline dry-run log\n";
    log << "timestamp: " << std::ctime(&now);
    log << "input_dir: " << cfg.input_dir << "\n";
    log << "output_dir: " << cfg.output_dir << "\n";
    log << "ops: ";
    for (size_t i = 0; i < cfg.ops.size(); ++i)
    {
        log << cfg.ops[i] << (i + 1 < cfg.ops.size() ? "," : "");
    }
    log << "\n";
    log << "batch: " << cfg.batch << "\n";
    log << "recursive: " << (cfg.recursive ? "true" : "false") << "\n";
    log << "files_found: " << files.size() << "\n";

    // Print a friendly summary to stdout too
    std::cout << "[OK] Found " << files.size() << " image(s). "
              << "Log: " << log_path << "\n";

    if (!cfg.dry_run)
    {
        std::cout << "(No processing yet — CUDA/NPP pipeline lands in Step 4/5)\n";
    }
    return 0;
}
