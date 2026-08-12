"""Throwaway check: confirm torch can actually use the GPU.

Reports the build variant, whether CUDA is visible, and benchmarks a real matrix
multiply on each device so a driver mismatch surfaces here rather than halfway
through a 5000-image run.

The first CUDA call also initialises the context and loads kernels, which can
cost more than the work itself. Timing it naively makes the GPU look barely
faster than the CPU, so each device is warmed up and then measured over several
repeats.

Usage:
    uv run python gpu_check.py
"""

import time

import torch

MATRIX_SIZE = 4096
WARMUP_RUNS = 3
TIMED_RUNS = 10


def benchmark(device: str) -> float:
    """Time one matrix multiply on a device, excluding start-up cost.

    Args:
        device: Torch device string, ``cpu`` or ``cuda``.

    Returns:
        float: Median seconds per multiply over the timed runs.
    """
    tensor = torch.randn(MATRIX_SIZE, MATRIX_SIZE, device=device)

    for _ in range(WARMUP_RUNS):
        tensor @ tensor
    if device == "cuda":
        torch.cuda.synchronize()

    timings = []
    for _ in range(TIMED_RUNS):
        start = time.perf_counter()
        tensor @ tensor
        if device == "cuda":
            torch.cuda.synchronize()
        timings.append(time.perf_counter() - start)

    timings.sort()
    return timings[len(timings) // 2]


def main() -> None:
    """Print the torch build and, if a GPU is present, benchmark against the CPU."""
    print(f"[*] torch version : {torch.__version__}")
    print(f"[*] built for CUDA: {torch.version.cuda or 'CPU-only build'}")
    print(f"[*] cuda available: {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        print("[-] torch cannot see a GPU. Everything will run on the CPU.")
        return

    print(f"[+] device        : {torch.cuda.get_device_name(0)}")
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"[+] total VRAM    : {total:.1f} GB")

    print(f"\n[*] Benchmarking a {MATRIX_SIZE}x{MATRIX_SIZE} matrix multiply")
    print(f"[*] {WARMUP_RUNS} warm-up run(s), median of {TIMED_RUNS} timed runs\n")

    cpu_time = benchmark("cpu")
    print(f"    cpu  : {cpu_time * 1000:8.1f} ms")

    gpu_time = benchmark("cuda")
    print(f"    cuda : {gpu_time * 1000:8.1f} ms")

    print(f"\n[+] GPU is {cpu_time / gpu_time:.1f}x faster on this workload.")
    print("[+] The pipeline steps detect the GPU automatically; no code change needed.")


if __name__ == "__main__":
    main()
