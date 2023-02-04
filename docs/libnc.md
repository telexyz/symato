https://bellard.org/libnc/libnc.html

LibNC is a C library for tensor manipulation. It supports automatic differentiation and can be used to implement machine learning models such as LSTM and Transformers. It has the following features:

- C API.
- Small library, no external dependency, available for Linux and Windows.
- Define-by-run automatic differentiation engine (same idea as PyTorch).
- High performance for both CPU (x86) and GPU (CUDA support). Optimized support of float32 and bfloat16 data types.
- CPU backend optimized for inference and small batch sizes.
- Optimized for online learning (i.e. simultaneous evaluation and training) using LSTM or Transformer models.
- Fully deterministic: return the same results at each run.
- Reproducible results (CPU backend only): return the same results regardless the CPU brand and OS.

LibNC requires an x86 CPU with AVX2 support.

The CUDA support is currently only available for Linux. CUDA version 11.x must be installed. Only Ampere GPUs are currently supported.

