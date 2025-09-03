#pragma once
#include <vector>
#include <string>
#include <fstream>
#include <cstdint>
#include <filesystem>
#include <sstream>

namespace fs = std::filesystem;

struct ImageU8
{
    int w = 0, h = 0;
    std::vector<uint8_t> data; // row-major, size = w*h
};

namespace detail
{
    // Read next token in a PGM header, skipping comments (# ...\n)
    inline bool next_token(std::istream &is, std::string &tok)
    {
        tok.clear();
        char c;
        // skip whitespace and comments
        while (is.get(c))
        {
            if (c == '#')
            {
                std::string dummy;
                std::getline(is, dummy); // skip rest of the line
                continue;
            }
            if (!std::isspace(static_cast<unsigned char>(c)))
            {
                tok.push_back(c);
                break;
            }
        }
        if (tok.empty())
            return false;
        // read until next whitespace
        while (is.get(c))
        {
            if (std::isspace(static_cast<unsigned char>(c)))
                break;
            tok.push_back(c);
        }
        return true;
    }
} // namespace detail

inline bool read_pgm(const fs::path &path, ImageU8 &out, std::string &err)
{
    err.clear();
    out = {};
    std::ifstream f(path, std::ios::binary);
    if (!f)
    {
        err = "cannot open file";
        return false;
    }

    std::string tok;
    if (!detail::next_token(f, tok))
    {
        err = "missing magic";
        return false;
    }
    bool ascii = false;
    if (tok == "P5")
        ascii = false;
    else if (tok == "P2")
        ascii = true;
    else
    {
        err = "unsupported magic (expect P5 or P2)";
        return false;
    }

    if (!detail::next_token(f, tok))
    {
        err = "missing width";
        return false;
    }
    int w = std::stoi(tok);
    if (!detail::next_token(f, tok))
    {
        err = "missing height";
        return false;
    }
    int h = std::stoi(tok);
    if (!detail::next_token(f, tok))
    {
        err = "missing maxval";
        return false;
    }
    int maxv = std::stoi(tok);
    if (w <= 0 || h <= 0 || maxv <= 0 || maxv > 255)
    {
        err = "bad dims/maxval";
        return false;
    }

    out.w = w;
    out.h = h;
    out.data.resize(w * h);

    if (ascii)
    {
        // P2: ASCII pixels separated by whitespace
        for (int i = 0; i < w * h; ++i)
        {
            if (!detail::next_token(f, tok))
            {
                err = "truncated P2 data";
                return false;
            }
            int v = std::stoi(tok);
            if (v < 0)
                v = 0;
            if (v > 255)
                v = 255;
            out.data[i] = static_cast<uint8_t>(v);
        }
    }
    else
    {
        // P5: after maxval there is one whitespace char, then raw bytes
        // consume one single whitespace if we ended right before pixel stream
        if (f.peek() == '\n' || f.peek() == '\r' || f.peek() == ' ' || f.peek() == '\t')
            f.get();
        f.read(reinterpret_cast<char *>(out.data.data()), out.data.size());
        if (f.gcount() != static_cast<std::streamsize>(out.data.size()))
        {
            err = "truncated P5 data";
            return false;
        }
    }
    return true;
}

inline bool write_pgm(const fs::path &path, const ImageU8 &img, std::string &err, bool binary = true)
{
    err.clear();
    if (img.w <= 0 || img.h <= 0 || static_cast<int>(img.data.size()) != img.w * img.h)
    {
        err = "invalid image";
        return false;
    }
    std::ofstream f(path, std::ios::binary);
    if (!f)
    {
        err = "cannot open for write";
        return false;
    }

    if (binary)
    {
        f << "P5\n"
          << img.w << " " << img.h << "\n255\n";
        f.write(reinterpret_cast<const char *>(img.data.data()), img.data.size());
    }
    else
    {
        f << "P2\n"
          << img.w << " " << img.h << "\n255\n";
        for (int i = 0; i < img.w * img.h; ++i)
        {
            f << int(img.data[i]) << ((i + 1) % img.w == 0 ? "\n" : " ");
        }
    }
    if (!f)
    {
        err = "write error";
        return false;
    }
    return true;
}
