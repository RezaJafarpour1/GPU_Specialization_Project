# GPU_Specialization_Project — CUDA Image Pipeline

## What this is (short description)
A CUDA-accelerated image processing pipeline that can process **hundreds of small or tens of large** grayscale images using **multi-stream GPU batching**. Implemented operations:
- `gauss` (5×5 separable Gaussian blur)
- `box` (3×3 separable box/mean filter)
- `sobel` (edge magnitude)
- `histeq` (histogram equalization)
- `invert` (CPU baseline)

The program reads **PGM (P5) grayscale** images and writes processed PGM outputs, plus **proof-of-execution artifacts** (`log.txt` and `metrics.csv`).

---

## Quickstart (Coursera GPU lab or any CUDA machine)

```bash
# build
make

# single-image, sequential path
./gpu_pipeline --input_dir data --output_dir outputs/run_seq --ops gauss,sobel,histeq --streams 1

# multi-stream throughput path (overlaps H2D/D2H and compute)
./gpu_pipeline --input_dir data --output_dir outputs/run_streams --ops gauss,sobel,histeq --streams 3
