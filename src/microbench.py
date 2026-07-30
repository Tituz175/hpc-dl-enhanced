"""Microbenchmark validation of the Roofline methodology (Decision #16).

Runs a small set of GPU kernels spanning a range of arithmetic intensities
on this workstation's RTX 3090, to construct a genuine MEASURED roofline
curve (achieved performance vs. operational intensity, both computed from
real timed kernels, not derived from a dataset's own reported counters)
and check that it takes the expected shape: low-intensity kernels tracking
the memory-bound ceiling, high-intensity kernels approaching the
compute-bound ceiling. This validates the Roofline METHODOLOGY used on
F-DATA in notebook 03, not the specific numbers — the RTX 3090 is neither
A64FX (F-DATA) nor Marconi100's V100s (PM100), so nothing here is a
hardware-matched ceiling for either production dataset (Decision #16's
explicit scope limitation).

Rodinia's own benchmark suite was not compiled/run directly here — four
standard kernels below span low to high arithmetic intensity the same way
Rodinia's suite does, following Konstantinidis et al. 2017's
microbenchmark-based Roofline approach in spirit rather than by running
their exact code, since building this self-contained substitute in
PyTorch (already installed, already CUDA-verified) was faster to build
and verify correct than standing up an external benchmark suite.
"""
from dataclasses import dataclass

import torch

# RTX 3090 published specs (NVIDIA's official spec sheet; GA102 die, 82
# SMs, boost clock 1.695 GHz): peak FP32 = 82 SMs * 128 FP32 cores/SM * 2
# FLOP/FMA * 1.695 GHz = 35.58 TFLOP/s. Consumer Ampere runs FP64 at 1/64
# the FP32 rate (a datacenter Ampere part like the A100 doesn't have this
# restriction) = 0.556 TFLOP/s. Memory: 24GB GDDR6X, 384-bit bus @ 19.5
# Gbps = 936.2 GB/s.
RTX3090_PEAK_FP32_FLOPS: float = 35.58e12
RTX3090_PEAK_FP64_FLOPS: float = 0.556e12
RTX3090_PEAK_BW_BYTES: float = 936.2e9


@dataclass
class KernelResult:
    name: str
    dtype: str
    n: int
    flops: float
    bytes_moved: float
    seconds: float

    @property
    def operational_intensity(self) -> float:
        return self.flops / self.bytes_moved

    @property
    def achieved_flops_per_sec(self) -> float:
        return self.flops / self.seconds


def _time_gpu(fn, n_iters: int = 20, n_warmup: int = 5) -> float:
    """Median-free mean GPU time per call, in seconds. Warmup iterations
    are discarded (first-call CUDA context/cache effects), and timing
    uses CUDA events rather than wall-clock so host-side Python overhead
    around the kernel launch isn't included."""
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(n_iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / 1000.0 / n_iters  # ms -> s, per call


def run_vector_add(n: int, dtype: torch.dtype) -> KernelResult:
    """y = a*x + y (SAXPY/DAXPY) — the lowest arithmetic intensity kernel
    here: two FLOPs per element against three elements' worth of memory
    traffic (read x, read y, write y)."""
    x = torch.randn(n, device="cuda", dtype=dtype)
    y = torch.randn(n, device="cuda", dtype=dtype)
    seconds = _time_gpu(lambda: y.add_(x, alpha=2.0))
    itemsize = x.element_size()
    flops = 2 * n
    bytes_moved = 3 * n * itemsize
    return KernelResult("vector_add", str(dtype), n, flops, bytes_moved, seconds)


def run_dot_product(n: int, dtype: torch.dtype) -> KernelResult:
    """Inner product of two vectors — same FLOP count as vector_add but
    one fewer array touched (no write-back of a full vector), so a
    slightly higher operational intensity."""
    x = torch.randn(n, device="cuda", dtype=dtype)
    z = torch.randn(n, device="cuda", dtype=dtype)
    seconds = _time_gpu(lambda: torch.dot(x, z))
    itemsize = x.element_size()
    flops = 2 * n
    bytes_moved = 2 * n * itemsize
    return KernelResult("dot_product", str(dtype), n, flops, bytes_moved, seconds)


def run_gemv(n: int, dtype: torch.dtype) -> KernelResult:
    """y = A @ x, A is n x n — reading the whole matrix once per call
    dominates memory traffic, giving a middling operational intensity
    between the vector kernels and GEMM below."""
    A = torch.randn(n, n, device="cuda", dtype=dtype)
    x = torch.randn(n, device="cuda", dtype=dtype)
    seconds = _time_gpu(lambda: torch.mv(A, x))
    itemsize = A.element_size()
    flops = 2 * n * n
    bytes_moved = (n * n + n) * itemsize
    return KernelResult("gemv", str(dtype), n, flops, bytes_moved, seconds)


def run_gemm(n: int, dtype: torch.dtype) -> KernelResult:
    """C = A @ B, all n x n — arithmetic intensity grows with n (FLOPs
    scale as n^3, naive memory traffic only as n^2), so this is the
    kernel that should approach the compute-bound ceiling at large n."""
    A = torch.randn(n, n, device="cuda", dtype=dtype)
    B = torch.randn(n, n, device="cuda", dtype=dtype)
    seconds = _time_gpu(lambda: A @ B)
    itemsize = A.element_size()
    flops = 2 * n ** 3
    bytes_moved = 3 * n * n * itemsize
    return KernelResult("gemm", str(dtype), n, flops, bytes_moved, seconds)
