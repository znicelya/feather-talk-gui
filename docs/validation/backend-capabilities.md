# Burn operator/backend gate

Burn is pinned to `=0.22.1`; committed Cargo.lock fixes Burn-family 0.22.1,
CubeCL 0.11.4 and CubeK 0.3.1. The patch-level CubeCL difference from the initial
0.11.1 reconnaissance is deliberate; actual builds use the locked sources.

CPU requests (`cpu:0`) and Flex requests (`flex:0`) both execute Burn Flex on the
host CPU. `cpu` is an application alias, not Burn's optional LLVM CPU backend.
No software GPU adapter substitutes for CPU. No AVX512 compiler target is forced.
Auto currently prefers Flex; otherwise it uses discovery order. This is a
conservative deterministic initial policy, not a performance ranking.

GPU IDs are backend enumeration indices and can change after hardware/driver
changes. Reports include Burn's actual device identity (part name and physical
PCI/LUID where exposed). Opening executes a tensor smoke check; only full
operator reports establish numerical parity. Missing references and unsupported
requests are errors. Actual training tensors use the selected Device.autodiff().

| Backend / hardware | Compiled | Executed | Numerical gate |
|---|---|---|---|
| Flex / i5-10400 host | Yes | Yes | 66/66 checks passed; 7/7 Rust tests |
| Vulkan / RTX2060 | Yes | Yes, physical identity confirmed | 66/66 checks passed |
| CUDA LLVM / RTX2060 | Yes | Yes, physical identity confirmed | 66/66 checks passed |
| Vulkan AMD / Intel Arc | Not tested | Not run | Not run |
| Metal Apple Silicon | Not tested | Not run | Not run |
| Intel Mac CPU | Not tested here | Not run | Not run |

Interpolation uses separable weighted select operations with additive-scatter
backward, avoiding the unsupported native GPU bilinear backward. BatchNorm uses
biased batch variance for normalization, unbiased variance for running updates,
and explicit train/eval plus batch counters. GroupNorm, pooling and attention
remain differentiable device tensor operations. Input/affine/attention parameter
gradients are checked with arbitrary upstream gradients, not output sums alone.

## Source-only normalization calibration

The initial prospective operator gate was atol=2e-6, rtol=2e-5. PyTorch's own
FP32 normalization backward differs from the original PyTorch module in FP64
using identical frozen FP32 inputs/upstream. The FP32 rerun reproduces every
original tensor exactly. Observed input-gradient error:

| Case | PyTorch FP32 vs FP64 max abs | Old gate failures | Fixed calibrated atol |
|---|---:|---:|---:|
| BatchNorm first | 2.91065337e-7 | 0/280 | 2e-6 |
| BatchNorm second | 6.44414518e-6 | 25/280 | 2.577658072133726e-5 |
| GroupNorm | 5.69306649e-6 | 70/280 | 2.2772265957371474e-5 |

Additional nonlinear cases independently calibrate to atol2e-6 for BatchNorm
(source error3.88245131e-7) and atol1.3292447604840163e-5 for GroupNorm
(source error3.32311190e-6). Both reject zero and1% scaled input gradients.

Controller ruling permits only normalization **input-gradient** calibration:
`max(original_atol, 4 * source_FP32_vs_FP64_error)`, keeping original rtol.
The factor is an empirical allowance for two FP32 reduction paths, not a proof.
Generator/manifest record per-tensor thresholds and full provenance. No Burn
result determines these thresholds; all outputs, affine gradients, running states
and other operators retain their original gates. All original tensor file hashes
remain unchanged.

The original linear input/upstream pairs make some normalization gradients tiny;
1% scaling can fall below their numerical noise. Two additional deterministic
sinusoidal-upstream cases with nondefault affine parameters verify useful input
and affine gradients, without changing the original RNG stream. Mutation tests
reject zero gradients and 1% scaling on each new case, and biased running variance
on the original BatchNorm state gate. Reference tests regenerate twice and verify
byte-for-byte reproducibility.

CUDA build uses process-only CUDARC_CUDA_VERSION=12060 and CUDA_PATH pointing to
the verified isolated CUDA12.6 NVCC redistribution containing libdevice. GPU
inference/training throughput, complete model parity and packaged platforms are
later gates; these operator results do not claim them.

Frozen convolution coverage is grouped Conv2d (groups2, dilation2) and depthwise
Conv1d (groups4, dilation2). Full audio TCN shapes/dilations1/2/4/8 and GELU
are required by the subsequent real audio model parity gate.

Final GPU runs on Windows used RTX2060 at PCI 0000:01:00.0: Vulkan
spirv_4318_7944 and CUDA ptx_sm75. Raw logs and per-comparison numeric errors are
in .artifacts/task3-logs/{vulkan,cuda}-final.log and
.artifacts/reports/operators-{vulkan,cuda}.json. Cold release compilation took
8m09s (Vulkan) and 10m25s (CUDA), not inference/training execution times.
The verified LLVM cache was consumed with process-only
TRACEL_LLVM_BUNDLER_SKIP_CHECKSUM_DOWNLOAD=1 after a sidecar download timeout;
archive and extracted-content checksum verification remained enabled.
