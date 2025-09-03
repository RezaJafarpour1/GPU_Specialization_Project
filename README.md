# GPU_Specialization_Project — CUDA Image Pipeline

A CUDA-accelerated image processing pipeline that can process **hundreds of small or tens of large** grayscale images using **multi-stream GPU batching**. Implemented operations:

- `gauss` — 5×5 separable Gaussian blur (CUDA)
- `box` — 3×3 separable mean filter (CUDA)
- `sobel` — edge magnitude (CUDA)
- `histeq` — histogram equalization (CUDA; device histogram + host LUT)
- `invert` — simple CPU baseline

The program reads **PGM (P5) grayscale** images and writes processed PGM outputs, and produces **proof-of-execution artifacts**:
- `log.txt` — run summary + (in streams mode) `throughput_images_per_sec`
- `metrics.csv` — per-image/per-op timings and a final `TOTAL_MS,all,<ms>` row

---

## Quickstart

```bash
# Build (Linux + CUDA)
make

# Sequential path (1 stream)
./gpu_pipeline --input_dir data --output_dir outputs/run_seq --ops gauss,sobel,histeq --streams 1

# Multi-stream throughput path (overlaps H2D/D2H and compute)
./gpu_pipeline --input_dir data --output_dir outputs/run_streams --ops gauss,sobel,histeq --streams 3
