## Roofline

```bash
══════════════════════════════════════════════════════
  Device 3: NVIDIA GeForce RTX 4090  (Compute 8.9)
══════════════════════════════════════════════════════

── SM count ────────────────────────────────────────────
  Multiprocessors : 128
  Max threads/SM  : 1536
  Max threads/blk : 1024
  Warp size       : 32

── Registers ──────────────────────────────────────────
  Per SM          : 65536
  Per thread block: 65536
  Per thread (SM) : 42
  Per thread (blk): 64

── Shared Memory ──────────────────────────────────────
  Per SM          : 100 KB
  Per thread block: 48 KB
  Per thread (SM) : 16.67 floats
  Per thread (blk): 12.00 floats

── Global Memory ───────────────────────────────────────
  Total           : 23.6 GB
  Memory clock    : 10501 MHz
  Bus width       : 384 bit
  Peak BW (theor) : 1008.1 GB/s

── Bandwidth (measured) ────────────────────────────────
  gmem read         ... 926.9 GB/s
  gmem write        ... 921.3 GB/s
  smem (baseline)   ... 19095.0 GB/s
  smem (ILP+sweep)  ... 22327.3 GB/s

── Tensor Core TF32 (measured) ─────────────────────────
  TF32 (ILP+sweep) ... 89.7 TFLOPS

── Tensor Core FP16 (measured) ─────────────────────────
  FP16 (ILP+sweep) ... 179.7 TFLOPS

── CUDA Core FP32 ───────────────────────────────────────
  Peak FP32 (theor): 82.6 TFLOPS
  FP32 (measured)   ... 78.8 TFLOPS

── Shared Memory BW ────────────────────────────────────
  Peak BW (theor)   : 41287.7 GB/s  (41.29 TB/s)
══════════════════════════════════════════════════════
```



## SGEMM

```bash
M=N=K=4096
warmup_iters=20, bench_iters=50

Baseline (cuBLAS SGEMM): avg_ms=2.759946, TFLOPS=49.797690

Kernel                  avg_ms      TFLOPS      vs_cublas   Result    max_abs     max_rel     
Naive                   25.265337   5.439823    0.109238    PASS      0.005371    0.000005    
Block_Tile              23.950745   5.738400    0.115234    PASS      0.005371    0.000005    
Thread_Tile             6.167304    22.285095   0.447513    PASS      0.005371    0.000005    
Vectorized_LDST         3.563659    38.566811   0.774470    PASS      0.005371    0.000005    
Warp_Tile               3.488628    39.396278   0.791127    PASS      0.005371    0.000005    
Double_Buffer           3.453706    39.794627   0.799126    PASS      0.005371    0.000005    
Tensor_Core_WMMA_HALF   1.664508    82.570306   1.658115    PASS      0.077026    0.000074    
Tensor_Core_WMMA_TF32   2.289107    60.040415   1.205687    PASS      0.779053    0.000732 
```



## FlashAttention

FlashAttention-V1 优化后在 RTX 4090 上性能结果

```bash
================ Attention Benchmark ================
batch=32, heads=16, N=1024, d=64
warmup_iters=10, bench_iters=10

baseline         avg=10.873606  ms
fa1_minimal      avg=203.226593 ms speedup_vs_baseline=0.053505  x correct=PASS max_abs=0.000001 max_rel=0.000003
fa1_version1     avg=266.535706 ms speedup_vs_baseline=0.040796  x correct=PASS max_abs=0.000001 max_rel=0.000003
fa1_version2     avg=62.314964  ms speedup_vs_baseline=0.174494  x correct=PASS max_abs=0.000001 max_rel=0.000003
fa1_version3     avg=9.209867   ms speedup_vs_baseline=1.180647  x correct=PASS max_abs=0.000027 max_rel=0.000054
=====================================================
```

```bash
================ Attention Benchmark ================
batch=32, heads=8, N=1024, d=128
warmup_iters=10, bench_iters=10

baseline         avg=6.240048   ms
fa1_minimal      avg=293.605591 ms speedup_vs_baseline=0.021253  x correct=PASS max_abs=0.000001 max_rel=0.000003
fa1_version1     avg=248.936356 ms speedup_vs_baseline=0.025067  x correct=PASS max_abs=0.000001 max_rel=0.000003
fa1_version2     avg=46.138466  ms speedup_vs_baseline=0.135246  x correct=PASS max_abs=0.000001 max_rel=0.000003
fa1_version3     avg=8.394333   ms speedup_vs_baseline=0.743364  x correct=PASS max_abs=0.000029 max_rel=0.000058
=====================================================
```



FlashAttention-V2 优化后在 RTX 4090 上性能结果

```bash
Config: batch=32, heads=8, N=1024, d=128, warmup=10, bench=50
Timing baseline: fp32 baseline on original fp32 inputs
Correctness ref for half kernels: baseline on the same half-quantized inputs

kernel          ref                   avg_ms     vs_base   correct       max_abs       max_rel
----------------------------------------------------------------------------------------------
baseline        self                  6.1948       1.000      PASS     0.000e+00     0.000e+00
fa1_version3    baseline_fp32         8.3360       0.743      PASS     2.885e-05     5.781e-05
fa2_version1    baseline_fp32        28.5878       0.217      PASS     1.907e-06     3.717e-06
fa2_version2    baseline_fp32        35.9479       0.172      PASS     1.878e-06     3.768e-06
fa2_version3    baseline_fp32        37.1951       0.167      PASS     1.907e-06     3.778e-06
fa2_version4    baseline_halfin       9.6331       0.643      PASS     3.461e-03     6.769e-03
fa2_version5    baseline_halfin       8.1408       0.761      PASS     3.461e-03     6.769e-03
fa2_version6    baseline_halfin       6.1644       1.005      PASS     2.853e-03     5.546e-03
fa2_version7    baseline_halfin       3.6332       1.705      PASS     2.853e-03     5.546e-03
```

```bash
Config: batch=2, heads=8, N=8192, d=128, warmup=10, bench=50
Timing baseline: fp32 baseline on original fp32 inputs
Correctness ref for half kernels: baseline on the same half-quantized inputs

kernel          ref                   avg_ms     vs_base   correct       max_abs       max_rel
----------------------------------------------------------------------------------------------
baseline        self                 24.4754       1.000      PASS     0.000e+00     0.000e+00
fa1_version3    baseline_fp32       355.1790       0.069      PASS     9.954e-06     2.003e-05
fa2_version1    baseline_fp32       114.3506       0.214      PASS     4.709e-06     9.295e-06
fa2_version2    baseline_fp32       142.8577       0.171      PASS     4.828e-06     9.611e-06
fa2_version3    baseline_fp32       149.5238       0.164      PASS     4.768e-06     9.530e-06
fa2_version4    baseline_halfin      40.8602       0.599      PASS     8.472e-03     1.695e-02
fa2_version5    baseline_halfin      34.1630       0.716      PASS     8.472e-03     1.695e-02
fa2_version6    baseline_halfin      24.4929       0.999      PASS     7.953e-03     1.599e-02
fa2_version7    baseline_halfin      14.1121       1.734      PASS     7.953e-03     1.599e-02
```



## Transpose

```bash
Naive transpose kernel execution time: 2.28488 ms
Shared transpose kernel 16 × 32 execution time: 0.953958 ms
Shared transpose kernel 16 × 32 (padding) execution time: 0.645626 ms
Shared transpose kernel 16 × 32 (swizzle) execution time: 0.64215 ms
Shared transpose kernel 32 × 16 (padding) execution time: 0.616448 ms
Shared transpose kernel 32 × 16 (swizzle) execution time: 0.609715 ms
```



## Softmax & Online Softmax

```bash
================ Benchmark Result ================
Shape: N = 128, C = 8192
Threads per block: 512
Warmup iterations: 20
Benchmark iterations: 200
CPU reference time: 8.366792 ms

Kernel              avg_ms        vs_best       vs_cpu      Result    max_abs_err   max_err_idx 
softmax_kernel_1    0.087765      0.551409      95.332149   PASS      0.000000      623211      
softmax_kernel_2    0.073103      0.661997      114.451537  PASS      0.000000      623211      
softmax_kernel_3    0.055501      0.871953      150.750396  PASS      0.000000      623211      
softmax_kernel_4    0.055992      0.864301      149.427492  PASS      0.000000      303152      
softmax_kernel_5    0.055868      0.866222      149.759577  PASS      0.000000      303152      
online_softmax1     0.063107      0.766864      132.581642  PASS      0.000000      622593      
online_softmax2     0.062130      0.778922      134.666418  PASS      0.000000      623211      
online_softmax3     0.048394      1.000000      172.888177  PASS      0.000000      623211      

Best kernel: online_softmax3 (0.048394 ms)
==================================================
```



## TopK

```bash
==================== TopK Benchmark ====================
name                              avg_ms        speedup_vs_base   correct   
baseline_cpu_partial_sort_copy    3.678655      1.000000          PASS      
Topk_Quick_Select                 50.667573     0.072604          PASS      
Topk_Priority_Queue               6.882297      0.534510          PASS      
Topk_Block_Reduce                 1.889132      1.947272          PASS      
Topk_Block_Reduce_Optimized       1.110400      3.312910          PASS      
========================================================
```

