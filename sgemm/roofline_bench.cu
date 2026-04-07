// roofline_bench.cu
#include <cuda_runtime.h>
#include <mma.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>

using namespace nvcuda;

// ─────────────────────────────────────────────────────────────────────────────
// helper
// ─────────────────────────────────────────────────────────────────────────────
#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error %s:%d  %s\n",                           \
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

// forward declaration
double measure_tc_tf32_tflops(int smCount, int maxThreadsPerSm);

// ─────────────────────────────────────────────────────────────────────────────
// gmem bandwidth benchmark
// ─────────────────────────────────────────────────────────────────────────────
static const size_t GMEM_N      = 1ULL << 26;   // 64M floats = 256 MB
static const int    GMEM_N4     = static_cast<int>(GMEM_N / 4); // 16M float4
static const int    GMEM_THRS   = 256;
static const int    GMEM_BLOCKS = 2048;
static const int    GMEM_REPS   = 20;

// Each thread accumulates float4 loads into a register.
// One scalar store per thread prevents DCE.
__global__ static void gmem_read_kernel(const float4* __restrict__ src,
                                        float* __restrict__ out,
                                        int n4) {
    int idx    = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    float acc = 0.f;
    for (int i = idx; i < n4; i += stride) {
        float4 v = src[i];
        acc += v.x + v.y + v.z + v.w;
    }
    out[idx] = acc;
}

// Each thread writes float4 values derived from thread index.
// This avoids extra global reads and prevents store elimination.
__global__ static void gmem_write_kernel(float4* __restrict__ dst, int n4) {
    int idx    = blockIdx.x * blockDim.x + threadIdx.x;
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
    float*  d_out = nullptr;

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

// ─────────────────────────────────────────────────────────────────────────────
// smem bandwidth benchmark
// ─────────────────────────────────────────────────────────────────────────────
static const int SMEM_REPS       = 512;
static const int SMEM_BLK_FLOATS = 1024; // 4KB per block

// Baseline smem benchmark.
// Important fix: initialize the whole shared-memory array, not only tid<256.
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
    const int blocks  = smCount * 4;
    const int threads = 256;
    const int REPS    = 500;

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

// ─────────────────────────────────────────────────────────────────────────────
// smem bandwidth benchmark (ILP + sweep)
// ─────────────────────────────────────────────────────────────────────────────
static const int SMEM_ILP_THREADS = 256;

// N_ACC independent accumulators.
// volatile prevents the compiler from hoisting / folding smem accesses.
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
    const int threads        = SMEM_ILP_THREADS;
    const int maxBlocksPerSm = maxThreadsPerSm / threads;
    const int REPS           = 500;

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

// ─────────────────────────────────────────────────────────────────────────────
// Tensor Core TF32 throughput benchmark
// Ampere and later (sm_80+), using WMMA 16x16x8 TF32 tiles
// ─────────────────────────────────────────────────────────────────────────────
static const int TC_THREADS = 128;  // 4 warps per block
static const int TC_INNER   = 1000; // mma_sync calls per warp per launch
static const int TC_OUTER   = 20;

static const int TC_M = 16;
static const int TC_N = 16;
static const int TC_K = 8;

static const long long TC_FLOPS_PER_MMA =
    2LL * TC_M * TC_N * TC_K; // 4096 FLOPs

template <int N_MMA>
__global__ static void tf32_mma_kernel(float* __restrict__ sink, int inner_reps) {
    wmma::fragment<wmma::matrix_a, TC_M, TC_N, TC_K,
                   wmma::precision::tf32, wmma::row_major> a;
    wmma::fragment<wmma::matrix_b, TC_M, TC_N, TC_K,
                   wmma::precision::tf32, wmma::col_major> b;
    wmma::fragment<wmma::accumulator, TC_M, TC_N, TC_K, float> c[N_MMA];

    wmma::fill_fragment(a, 1.0f);
    wmma::fill_fragment(b, 1.0f);

    #pragma unroll
    for (int n = 0; n < N_MMA; ++n) {
        wmma::fill_fragment(c[n], 0.0f);
    }

    for (int r = 0; r < inner_reps; ++r) {
        #pragma unroll
        for (int n = 0; n < N_MMA; ++n) {
            wmma::mma_sync(c[n], a, b, c[n]);
        }
    }

    float total = 0.f;
    #pragma unroll
    for (int n = 0; n < N_MMA; ++n) {
        for (int i = 0; i < c[n].num_elements; ++i) {
            total += c[n].x[i];
        }
    }

    if (total == -1.f) sink[blockIdx.x] = total;
}

template <int N_MMA>
static double tc_tf32_one(int smCount,
                          int maxThreadsPerSm,
                          cudaEvent_t t0,
                          cudaEvent_t t1) {
    const int warps_per_block   = TC_THREADS / 32;
    const int max_blocks_per_sm = maxThreadsPerSm / TC_THREADS;

    double best = 0.0;

    for (int bps = 1; bps <= max_blocks_per_sm; ++bps) {
        const int blocks = smCount * bps;

        float* d_sink = nullptr;
        CUDA_CHECK(cudaMalloc(&d_sink, static_cast<size_t>(blocks) * sizeof(float)));

        tf32_mma_kernel<N_MMA><<<blocks, TC_THREADS>>>(d_sink, TC_INNER);
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(t0));
        for (int r = 0; r < TC_OUTER; ++r) {
            tf32_mma_kernel<N_MMA><<<blocks, TC_THREADS>>>(d_sink, TC_INNER);
        }
        CUDA_CHECK(cudaEventRecord(t1));
        CUDA_CHECK(cudaEventSynchronize(t1));

        float ms = 0.f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));

        double flops =
            static_cast<double>(blocks) *
            warps_per_block *
            N_MMA *
            TC_INNER *
            TC_FLOPS_PER_MMA *
            TC_OUTER;

        best = std::max(best, flops / 1e12 / (ms / 1e3));

        CUDA_CHECK(cudaFree(d_sink));
    }

    return best;
}

double measure_tc_tf32_tflops(int smCount, int maxThreadsPerSm) {
    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));

    double best = 0.0;
    best = std::max(best, tc_tf32_one<1>(smCount, maxThreadsPerSm, t0, t1));
    best = std::max(best, tc_tf32_one<2>(smCount, maxThreadsPerSm, t0, t1));
    best = std::max(best, tc_tf32_one<4>(smCount, maxThreadsPerSm, t0, t1));
    best = std::max(best, tc_tf32_one<8>(smCount, maxThreadsPerSm, t0, t1));

    CUDA_CHECK(cudaEventDestroy(t0));
    CUDA_CHECK(cudaEventDestroy(t1));
    return best;
}

// ─────────────────────────────────────────────────────────────────────────────
// public API
// ─────────────────────────────────────────────────────────────────────────────
void print_device_info(int dev) {
    cudaDeviceProp p;
    CUDA_CHECK(cudaGetDeviceProperties(&p, dev));
    CUDA_CHECK(cudaSetDevice(dev));

    double peak_gmem_GBs =
        static_cast<double>(p.memoryClockRate) * 1e3 *
        (p.memoryBusWidth / 8.0) *
        2.0 / 1e9;

    printf("══════════════════════════════════════════════════════\n");
    printf("  Device %d: %s  (Compute %d.%d)\n", dev, p.name, p.major, p.minor);
    printf("══════════════════════════════════════════════════════\n");

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

    if (p.major >= 8) {
        printf("\n── Tensor Core TF32 (measured) ─────────────────────────\n");
        printf("  TF32 (ILP+sweep) ... "); fflush(stdout);
        printf("%.1f TFLOPS\n",
               measure_tc_tf32_tflops(p.multiProcessorCount,
                                      p.maxThreadsPerMultiProcessor));
    } else {
        printf("\n── Tensor Core TF32 (measured) ─────────────────────────\n");
        printf("  skipped: TF32 WMMA requires Ampere or newer GPU\n");
    }

    printf("══════════════════════════════════════════════════════\n\n");
}

// ─────────────────────────────────────────────────────────────────────────────
// main
// ─────────────────────────────────────────────────────────────────────────────
int main(int argc, char** argv) {
    CUDA_CHECK(cudaSetDevice(3));

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