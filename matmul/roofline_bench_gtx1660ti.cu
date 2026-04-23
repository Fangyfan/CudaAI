#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error %s:%d  %s\n",                         \
                    __FILE__, __LINE__, cudaGetErrorString(err));               \
            std::exit(EXIT_FAILURE);                                            \
        }                                                                       \
    } while (0)

template <typename F>
double run_timed(cudaEvent_t t0, cudaEvent_t t1, F launch, int reps) {
    launch();
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(t0));
    for (int r = 0; r < reps; ++r) launch();
    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));

    float ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
    return static_cast<double>(ms);
}

static int cuda_cores_per_sm(int major, int minor) {
    const int v = major * 10 + minor;
    switch (v) {
        case 89: return 128; // Ada Lovelace, e.g. RTX 4090
        case 86: return 128; // Ampere GA10x
        case 80: return 64;  // A100
        case 75: return 64;  // Turing
        case 70: return 64;  // Volta
        default: return 0;
    }
}

static bool is_gtx_1660_ti(const cudaDeviceProp& p) {
    return std::strstr(p.name, "1660 Ti") != nullptr;
}

static double estimate_cuda_core_fp32_peak_tflops(const cudaDeviceProp& p) {
    int cores_per_sm = cuda_cores_per_sm(p.major, p.minor);
    if (cores_per_sm == 0) return 0.0;

    double sm_clock_ghz = static_cast<double>(p.clockRate) * 1e-6;
    double total_cores = static_cast<double>(cores_per_sm) * p.multiProcessorCount;
    return total_cores * 2.0 * sm_clock_ghz / 1e3;
}

static double estimate_cuda_core_fp16_peak_tflops_gtx1660ti(const cudaDeviceProp& p) {
    double fp32_peak = estimate_cuda_core_fp32_peak_tflops(p);
    if (fp32_peak == 0.0) return 0.0;
    return fp32_peak * 2.0;
}

static double estimate_smem_peak_GBs(const cudaDeviceProp& p) {
    double sm_clock_hz = static_cast<double>(p.clockRate) * 1e3;
    double bytes_per_sm_per_cycle = 32.0 * 4.0;
    return bytes_per_sm_per_cycle * p.multiProcessorCount * sm_clock_hz / 1e9;
}

double measure_cuda_core_fp32_tflops(int smCount, int maxThreadsPerSm);
double measure_cuda_core_fp16_tflops(int smCount, int maxThreadsPerSm);

static const size_t GMEM_N = 1ULL << 26;     // 64M floats = 256 MB
static const int GMEM_N4 = static_cast<int>(GMEM_N / 4);
static const int GMEM_THRS = 256;
static const int GMEM_BLOCKS = 2048;
static const int GMEM_REPS = 20;

__global__ static void gmem_read_kernel(const float4* __restrict__ src,
                                        float* __restrict__ out,
                                        int n4) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    float acc = 0.f;
    for (int i = idx; i < n4; i += stride) {
        float4 v = src[i];
        acc += v.x + v.y + v.z + v.w;
    }
    out[idx] = acc;
}

__global__ static void gmem_write_kernel(float4* __restrict__ dst, int n4) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    float4 v = make_float4(static_cast<float>(idx),
                           static_cast<float>(idx) + 0.1f,
                           static_cast<float>(idx) + 0.2f,
                           static_cast<float>(idx) + 0.3f);

    for (int i = idx; i < n4; i += stride) {
        dst[i] = v;
    }
}

static double measure_gmem_read_bandwidth_GBs() {
    const size_t src_bytes = GMEM_N * sizeof(float);
    const size_t out_bytes = static_cast<size_t>(GMEM_BLOCKS) * GMEM_THRS * sizeof(float);

    float4* d_src = nullptr;
    float* d_out = nullptr;

    CUDA_CHECK(cudaMalloc(&d_src, src_bytes));
    CUDA_CHECK(cudaMalloc(&d_out, out_bytes));
    CUDA_CHECK(cudaMemset(d_src, 0, src_bytes));

    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));

    auto launch = [&] {
        gmem_read_kernel<<<GMEM_BLOCKS, GMEM_THRS>>>(d_src, d_out, GMEM_N4);
    };
    double ms = run_timed(t0, t1, launch, GMEM_REPS);

    CUDA_CHECK(cudaEventDestroy(t0));
    CUDA_CHECK(cudaEventDestroy(t1));
    CUDA_CHECK(cudaFree(d_src));
    CUDA_CHECK(cudaFree(d_out));

    return static_cast<double>(src_bytes) * GMEM_REPS / 1e9 / (ms / 1e3);
}

static double measure_gmem_write_bandwidth_GBs() {
    const size_t dst_bytes = GMEM_N * sizeof(float);

    float4* d_dst = nullptr;
    CUDA_CHECK(cudaMalloc(&d_dst, dst_bytes));

    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));

    auto launch = [&] {
        gmem_write_kernel<<<GMEM_BLOCKS, GMEM_THRS>>>(d_dst, GMEM_N4);
    };
    double ms = run_timed(t0, t1, launch, GMEM_REPS);

    CUDA_CHECK(cudaEventDestroy(t0));
    CUDA_CHECK(cudaEventDestroy(t1));
    CUDA_CHECK(cudaFree(d_dst));

    return static_cast<double>(dst_bytes) * GMEM_REPS / 1e9 / (ms / 1e3);
}

static const int SMEM_REPS = 512;
static const int SMEM_BLK_FLOATS = 1024;

__global__ static void smem_bw_kernel(float* sink) {
    __shared__ float smem[SMEM_BLK_FLOATS];

    int tid = threadIdx.x;

    for (int i = tid; i < SMEM_BLK_FLOATS; i += blockDim.x) {
        smem[i] = static_cast<float>(i);
    }
    __syncthreads();

    float acc = 0.f;
    for (int r = 0; r < SMEM_REPS; ++r) {
        for (int i = tid; i < SMEM_BLK_FLOATS; i += blockDim.x) {
            acc += smem[i];
        }
    }

    if (acc == -1.f) sink[blockIdx.x] = acc;
}

static double measure_smem_bandwidth_GBs(int smCount) {
    const int blocks = smCount * 4;
    const int threads = 256;
    const int REPS = 500;

    float* d_sink = nullptr;
    CUDA_CHECK(cudaMalloc(&d_sink, static_cast<size_t>(blocks) * sizeof(float)));

    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));

    auto launch = [&] {
        smem_bw_kernel<<<blocks, threads>>>(d_sink);
    };
    double ms = run_timed(t0, t1, launch, REPS);

    CUDA_CHECK(cudaEventDestroy(t0));
    CUDA_CHECK(cudaEventDestroy(t1));
    CUDA_CHECK(cudaFree(d_sink));

    double bytes_per_launch =
        static_cast<double>(blocks) * SMEM_BLK_FLOATS * SMEM_REPS * sizeof(float);
    double total_GB = bytes_per_launch * REPS / 1e9;
    return total_GB / (ms / 1e3);
}

static const int SMEM_ILP_THREADS = 256;

template <int N_ACC>
__global__ static void smem_bw_ilp_kernel(float* sink) {
    __shared__ float smem[N_ACC * SMEM_ILP_THREADS];

    int tid = threadIdx.x;

    for (int k = 0; k < N_ACC; ++k) {
        smem[tid + k * SMEM_ILP_THREADS] =
            static_cast<float>(tid + k * SMEM_ILP_THREADS);
    }
    __syncthreads();

    volatile float* vs = smem;

    float acc[N_ACC];
    #pragma unroll
    for (int k = 0; k < N_ACC; ++k) acc[k] = 0.f;

    for (int r = 0; r < SMEM_REPS; ++r) {
        #pragma unroll
        for (int k = 0; k < N_ACC; ++k) {
            acc[k] += vs[tid + k * SMEM_ILP_THREADS];
        }
    }

    float total = 0.f;
    #pragma unroll
    for (int k = 0; k < N_ACC; ++k) total += acc[k];

    if (total == -1.f) sink[blockIdx.x] = total;
}

template <int N_ACC>
static double measure_smem_ilp_one(int smCount,
                                   int maxThreadsPerSm,
                                   cudaEvent_t t0,
                                   cudaEvent_t t1) {
    const int threads = SMEM_ILP_THREADS;
    const int maxBlocksPerSm = maxThreadsPerSm / threads;
    const int REPS = 500;

    double best = 0.0;

    for (int bps = 1; bps <= maxBlocksPerSm; ++bps) {
        const int blocks = smCount * bps;

        float* d_sink = nullptr;
        CUDA_CHECK(cudaMalloc(&d_sink, static_cast<size_t>(blocks) * sizeof(float)));

        auto launch = [&] {
            smem_bw_ilp_kernel<N_ACC><<<blocks, threads>>>(d_sink);
        };
        double ms = run_timed(t0, t1, launch, REPS);

        double bytes =
            static_cast<double>(N_ACC) * threads * blocks * SMEM_REPS * sizeof(float);
        double bw = bytes * REPS / 1e9 / (ms / 1e3);

        best = std::max(best, bw);
        CUDA_CHECK(cudaFree(d_sink));
    }
    return best;
}

static double measure_smem_ilp_bandwidth_GBs(int smCount, int maxThreadsPerSm) {
    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));

    double best = 0.0;
    best = std::max(best, measure_smem_ilp_one<1 >(smCount, maxThreadsPerSm, t0, t1));
    best = std::max(best, measure_smem_ilp_one<2 >(smCount, maxThreadsPerSm, t0, t1));
    best = std::max(best, measure_smem_ilp_one<4 >(smCount, maxThreadsPerSm, t0, t1));
    best = std::max(best, measure_smem_ilp_one<8 >(smCount, maxThreadsPerSm, t0, t1));
    best = std::max(best, measure_smem_ilp_one<16>(smCount, maxThreadsPerSm, t0, t1));

    CUDA_CHECK(cudaEventDestroy(t0));
    CUDA_CHECK(cudaEventDestroy(t1));
    return best;
}

static const int CC_THREADS = 256;
static const int CC_INNER = 4096;
static const int CC_OUTER = 20;

template <int N_ACC>
__global__ static void cuda_core_fp32_kernel(float* __restrict__ sink, int inner_reps) {
    float acc[N_ACC], a[N_ACC], b[N_ACC];

    #pragma unroll
    for (int i = 0; i < N_ACC; ++i) {
        acc[i] = 0.1f * (i + 1);
        a[i] = 1.0000001f + 0.0001f * (threadIdx.x & 7) + i;
        b[i] = 1.0000002f + 0.0002f * ((threadIdx.x >> 3) & 7) + i;
    }

    for (int r = 0; r < inner_reps; ++r) {
        #pragma unroll
        for (int i = 0; i < N_ACC; ++i) {
            acc[i] += a[i] * b[i];
        }
    }

    float total = 0.f;
    #pragma unroll
    for (int i = 0; i < N_ACC; ++i) total += acc[i];

    if (total == -1.f) sink[blockIdx.x] = total;
}

template <int N_ACC>
static double cuda_core_fp32_one(int smCount,
                                 int maxThreadsPerSm,
                                 cudaEvent_t t0,
                                 cudaEvent_t t1) {
    const int threads = CC_THREADS;
    const int max_blocks_per_sm = maxThreadsPerSm / threads;

    double best = 0.0;

    for (int bps = 1; bps <= max_blocks_per_sm; ++bps) {
        const int blocks = smCount * bps;

        float* d_sink = nullptr;
        CUDA_CHECK(cudaMalloc(&d_sink, static_cast<size_t>(blocks) * sizeof(float)));

        cuda_core_fp32_kernel<N_ACC><<<blocks, threads>>>(d_sink, CC_INNER);
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(t0));
        for (int r = 0; r < CC_OUTER; ++r) {
            cuda_core_fp32_kernel<N_ACC><<<blocks, threads>>>(d_sink, CC_INNER);
        }
        CUDA_CHECK(cudaEventRecord(t1));
        CUDA_CHECK(cudaEventSynchronize(t1));

        float ms = 0.f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));

        double flops =
            static_cast<double>(blocks) *
            threads *
            N_ACC *
            CC_INNER *
            2.0 *
            CC_OUTER;

        double tflops = flops / 1e12 / (ms / 1e3);
        best = std::max(best, tflops);

        CUDA_CHECK(cudaFree(d_sink));
    }

    return best;
}

double measure_cuda_core_fp32_tflops(int smCount, int maxThreadsPerSm) {
    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));

    double best = 0.0;
    best = std::max(best, cuda_core_fp32_one<1 >(smCount, maxThreadsPerSm, t0, t1));
    best = std::max(best, cuda_core_fp32_one<2 >(smCount, maxThreadsPerSm, t0, t1));
    best = std::max(best, cuda_core_fp32_one<4 >(smCount, maxThreadsPerSm, t0, t1));
    best = std::max(best, cuda_core_fp32_one<8 >(smCount, maxThreadsPerSm, t0, t1));
    best = std::max(best, cuda_core_fp32_one<16>(smCount, maxThreadsPerSm, t0, t1));

    CUDA_CHECK(cudaEventDestroy(t0));
    CUDA_CHECK(cudaEventDestroy(t1));
    return best;
}

template <int N_ACC>
__global__ static void cuda_core_fp16_kernel(float* __restrict__ sink, int inner_reps) {
    __half2 acc[N_ACC], a[N_ACC], b[N_ACC];

    #pragma unroll
    for (int i = 0; i < N_ACC; ++i) {
        float ax = 1.0000f + 0.0010f * (threadIdx.x & 7) + 0.0100f * i;
        float ay = 1.1250f + 0.0015f * (threadIdx.x & 3) + 0.0200f * i;
        float bx = 1.2500f + 0.0005f * ((threadIdx.x >> 3) & 7) + 0.0150f * i;
        float by = 1.3750f + 0.0007f * ((threadIdx.x >> 2) & 7) + 0.0250f * i;

        acc[i] = __floats2half2_rn(0.1f * (i + 1), 0.2f * (i + 1));
        a[i] = __floats2half2_rn(ax, ay);
        b[i] = __floats2half2_rn(bx, by);
    }

    for (int r = 0; r < inner_reps; ++r) {
        #pragma unroll
        for (int i = 0; i < N_ACC; ++i) {
            acc[i] = __hfma2(a[i], b[i], acc[i]);
        }
    }

    float total = 0.f;
    #pragma unroll
    for (int i = 0; i < N_ACC; ++i) {
        float2 v = __half22float2(acc[i]);
        total += v.x + v.y;
    }

    if (total == -1.f) sink[blockIdx.x] = total;
}

template <int N_ACC>
static double cuda_core_fp16_one(int smCount,
                                 int maxThreadsPerSm,
                                 cudaEvent_t t0,
                                 cudaEvent_t t1) {
    const int threads = CC_THREADS;
    const int max_blocks_per_sm = maxThreadsPerSm / threads;

    double best = 0.0;

    for (int bps = 1; bps <= max_blocks_per_sm; ++bps) {
        const int blocks = smCount * bps;

        float* d_sink = nullptr;
        CUDA_CHECK(cudaMalloc(&d_sink, static_cast<size_t>(blocks) * sizeof(float)));

        cuda_core_fp16_kernel<N_ACC><<<blocks, threads>>>(d_sink, CC_INNER);
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(t0));
        for (int r = 0; r < CC_OUTER; ++r) {
            cuda_core_fp16_kernel<N_ACC><<<blocks, threads>>>(d_sink, CC_INNER);
        }
        CUDA_CHECK(cudaEventRecord(t1));
        CUDA_CHECK(cudaEventSynchronize(t1));

        float ms = 0.f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));

        double flops =
            static_cast<double>(blocks) *
            threads *
            N_ACC *
            CC_INNER *
            4.0 *
            CC_OUTER;

        double tflops = flops / 1e12 / (ms / 1e3);
        best = std::max(best, tflops);

        CUDA_CHECK(cudaFree(d_sink));
    }

    return best;
}

double measure_cuda_core_fp16_tflops(int smCount, int maxThreadsPerSm) {
    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));

    double best = 0.0;
    best = std::max(best, cuda_core_fp16_one<1 >(smCount, maxThreadsPerSm, t0, t1));
    best = std::max(best, cuda_core_fp16_one<2 >(smCount, maxThreadsPerSm, t0, t1));
    best = std::max(best, cuda_core_fp16_one<4 >(smCount, maxThreadsPerSm, t0, t1));
    best = std::max(best, cuda_core_fp16_one<8 >(smCount, maxThreadsPerSm, t0, t1));
    best = std::max(best, cuda_core_fp16_one<16>(smCount, maxThreadsPerSm, t0, t1));

    CUDA_CHECK(cudaEventDestroy(t0));
    CUDA_CHECK(cudaEventDestroy(t1));
    return best;
}

void print_device_info(int dev) {
    cudaDeviceProp p;
    CUDA_CHECK(cudaGetDeviceProperties(&p, dev));
    CUDA_CHECK(cudaSetDevice(dev));

    double peak_gmem_GBs =
        static_cast<double>(p.memoryClockRate) * 1e3 *
        (p.memoryBusWidth / 8.0) *
        2.0 / 1e9;
    double peak_cuda_core_fp32_tflops = estimate_cuda_core_fp32_peak_tflops(p);
    double peak_cuda_core_fp16_tflops = estimate_cuda_core_fp16_peak_tflops_gtx1660ti(p);
    double peak_smem_GBs = estimate_smem_peak_GBs(p);

    printf("══════════════════════════════════════════════════════\n");
    printf("  Device %d: %s  (Compute %d.%d)\n", dev, p.name, p.major, p.minor);
    printf("══════════════════════════════════════════════════════\n");

    if (!is_gtx_1660_ti(p)) {
        printf("\n[warning] This code is tuned for NVIDIA GeForce GTX 1660 Ti style roofline evaluation.\n");
        printf("          It will still run on other GPUs, but the printed FP16 interpretation is\n");
        printf("          intended for GTX 1660 Ti / TU116 class devices.\n");
    }

    printf("\n── SM count ────────────────────────────────────────────\n");
    printf("  Multiprocessors : %d\n", p.multiProcessorCount);
    printf("  Max threads/SM  : %d\n", p.maxThreadsPerMultiProcessor);
    printf("  Max threads/blk : %d\n", p.maxThreadsPerBlock);
    printf("  Warp size       : %d\n", p.warpSize);

    printf("\n── Registers ──────────────────────────────────────────\n");
    printf("  Per SM          : %d\n", p.regsPerMultiprocessor);
    printf("  Per thread block: %d\n", p.regsPerBlock);
    printf("  Per thread (SM) : %d\n",
           p.regsPerMultiprocessor / p.maxThreadsPerMultiProcessor);
    printf("  Per thread (blk): %d\n",
           p.regsPerBlock / p.maxThreadsPerBlock);

    printf("\n── Shared Memory ──────────────────────────────────────\n");
    printf("  Per SM          : %zu KB\n", p.sharedMemPerMultiprocessor / 1024);
    printf("  Per thread block: %zu KB\n", p.sharedMemPerBlock / 1024);
    printf("  Per thread (SM) : %.2f floats\n",
           static_cast<double>(p.sharedMemPerMultiprocessor) / sizeof(float) /
           p.maxThreadsPerMultiProcessor);
    printf("  Per thread (blk): %.2f floats\n",
           static_cast<double>(p.sharedMemPerBlock) / sizeof(float) /
           p.maxThreadsPerBlock);

    printf("\n── Global Memory ───────────────────────────────────────\n");
    printf("  Total           : %.1f GB\n",
           static_cast<double>(p.totalGlobalMem) / (1ULL << 30));
    printf("  Memory clock    : %.0f MHz\n", p.memoryClockRate * 1e-3);
    printf("  Bus width       : %d bit\n", p.memoryBusWidth);
    printf("  Peak BW (theor) : %.1f GB/s\n", peak_gmem_GBs);

    printf("\n── Bandwidth (measured) ────────────────────────────────\n");
    printf("  gmem read         ... "); fflush(stdout);
    printf("%.1f GB/s\n", measure_gmem_read_bandwidth_GBs());

    printf("  gmem write        ... "); fflush(stdout);
    printf("%.1f GB/s\n", measure_gmem_write_bandwidth_GBs());

    printf("  smem (baseline)   ... "); fflush(stdout);
    printf("%.1f GB/s\n", measure_smem_bandwidth_GBs(p.multiProcessorCount));

    printf("  smem (ILP+sweep)  ... "); fflush(stdout);
    printf("%.1f GB/s\n",
           measure_smem_ilp_bandwidth_GBs(p.multiProcessorCount,
                                          p.maxThreadsPerMultiProcessor));

    printf("\n── Tensor Core TF32 (removed for GTX 1660 Ti) ──────────\n");
    printf("  skipped: TF32 is an Ampere Tensor Core path, not applicable to GTX 1660 Ti\n");

    printf("\n── Tensor Core FP16 (removed for GTX 1660 Ti) ──────────\n");
    printf("  skipped: GTX 1660 Ti does not expose the Tensor Core path used by the 4090 code\n");

    printf("\n── CUDA Core FP32 ───────────────────────────────────────\n");
    if (peak_cuda_core_fp32_tflops > 0.0) {
        printf("  Peak FP32 (theor): %.2f TFLOPS\n", peak_cuda_core_fp32_tflops);
    } else {
        printf("  Peak FP32 (theor): unknown arch mapping\n");
    }
    printf("  FP32 (measured)   ... "); fflush(stdout);
    printf("%.2f TFLOPS\n",
           measure_cuda_core_fp32_tflops(p.multiProcessorCount,
                                         p.maxThreadsPerMultiProcessor));

    printf("\n── CUDA Core FP16 ───────────────────────────────────────\n");
    if (is_gtx_1660_ti(p) && peak_cuda_core_fp16_tflops > 0.0) {
        printf("  Peak FP16 (theor): %.2f TFLOPS\n", peak_cuda_core_fp16_tflops);
        printf("  Note             : GTX 1660 Ti should be interpreted as FP16 on CUDA cores,\n");
        printf("                     not FP16 on Tensor Cores\n");
    } else if (peak_cuda_core_fp16_tflops > 0.0) {
        printf("  Peak FP16 (theor): %.2f TFLOPS (using GTX 1660 Ti style 2x FP32 assumption)\n",
               peak_cuda_core_fp16_tflops);
    } else {
        printf("  Peak FP16 (theor): unavailable\n");
    }
    printf("  FP16 (measured)   ... "); fflush(stdout);
    printf("%.2f TFLOPS\n",
           measure_cuda_core_fp16_tflops(p.multiProcessorCount,
                                         p.maxThreadsPerMultiProcessor));

    printf("\n── Shared Memory BW ────────────────────────────────────\n");
    printf("  Peak BW (theor)   : %.1f GB/s  (%.2f TB/s)\n",
           peak_smem_GBs, peak_smem_GBs / 1e3);

    printf("══════════════════════════════════════════════════════\n\n");
}

int main(int argc, char** argv) {
    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));

    if (device_count == 0) {
        fprintf(stderr, "No CUDA device found.\n");
        return EXIT_FAILURE;
    }

    int dev = 0;
    if (argc >= 2) {
        dev = std::atoi(argv[1]);
    }

    if (dev < 0 || dev >= device_count) {
        fprintf(stderr, "Invalid device id %d, valid range is [0, %d).\n",
                dev, device_count);
        return EXIT_FAILURE;
    }

    print_device_info(dev);
    return 0;
}
