#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <cub/block/block_reduce.cuh>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err)             \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl;   \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)

void softmax_forward_cpu(float* out, const float* in, int N, int C) {
    for (int i = 0; i < N; ++i) {
        const float* in_row = in + i * C;
        float* out_row = out + i * C;

        float max_val = -INFINITY;
        for (int j = 0; j < C; ++j) {
            if (max_val < in_row[j]) {
                max_val = in_row[j];
            }
        }

        float sum = 0.0f;
        for (int j = 0; j < C; ++j) {
            out_row[j] = std::exp(in_row[j] - max_val);
            sum += out_row[j];
        }

        float norm = 1.0f / sum;
        for (int j = 0; j < C; ++j) {
            out_row[j] *= norm;
        }
    }
}

namespace softmax {
template <int BLOCK_DIM>
__global__ void softmax_kernel_1(float* out, const float* in, int C) {
    using BlockReduce = cub::BlockReduce<float, BLOCK_DIM>;
    __shared__ typename BlockReduce::TempStorage temp;
    __shared__ float shared_val;

    const float* in_row = in + blockIdx.x * C;
    float* out_row = out + blockIdx.x * C;

    float max_val = -INFINITY;
    for (int i = threadIdx.x; i < C; i += blockDim.x) {
        max_val = fmaxf(max_val, in_row[i]);
    }
    max_val = BlockReduce(temp).Reduce(max_val, cub::Max());
    if (threadIdx.x == 0) {
        shared_val = max_val;
    }
    __syncthreads();
    max_val = shared_val;

    float sum = 0.0f;
    for (int i = threadIdx.x; i < C; i += blockDim.x) {
        float exp_val = __expf(in_row[i] - max_val);
        sum += exp_val;
        out_row[i] = exp_val;
    }
    sum = BlockReduce(temp).Reduce(sum, cub::Sum());
    if (threadIdx.x == 0) {
        shared_val = sum;
    }
    __syncthreads();
    sum = shared_val;

    for (int i = threadIdx.x; i < C; i += blockDim.x) {
        out_row[i] /= sum;
    }
}

template <int BLOCK_DIM>
__global__ void softmax_kernel_2(float* out, const float* in, int C) {
    using BlockReduce = cub::BlockReduce<float, BLOCK_DIM>;
    __shared__ typename BlockReduce::TempStorage temp;
    __shared__ float shared_val;

    const float* in_row = in + blockIdx.x * C;
    float* out_row = out + blockIdx.x * C;

    float max_val = -INFINITY;
    for (int i = threadIdx.x; i < C; i += blockDim.x) {
        max_val = fmaxf(max_val, in_row[i]);
    }
    max_val = BlockReduce(temp).Reduce(max_val, cub::Max());
    if (threadIdx.x == 0) {
        shared_val = max_val;
    }
    __syncthreads();
    max_val = shared_val;

    float sum = 0.0f;
    for (int i = threadIdx.x; i < C; i += blockDim.x) {
        sum += __expf(in_row[i] - max_val);
    }
    sum = BlockReduce(temp).Reduce(sum, cub::Sum());
    if (threadIdx.x == 0) {
        shared_val = sum;
    }
    __syncthreads();
    sum = shared_val;

    for (int i = threadIdx.x; i < C; i += blockDim.x) {
        out_row[i] = __expf(in_row[i] - max_val) / sum;
    }
}
}  // namespace softmax

namespace online_softmax1 {
struct __align__(8) MD {
    float m;
    float d;
};

struct MD_OP {
    __device__ __forceinline__ MD operator()(const MD& a, const MD& b) const {
        MD res;
        res.m = fmaxf(a.m, b.m);
        res.d = a.d * __expf(a.m - res.m) + b.d * __expf(b.m - res.m);
        return res;
    }
};

template <int BLOCK_DIM>
__global__ void online_softmax_kernel(const float* __restrict__ input, float* __restrict__ output, int C) {
    input += blockIdx.x * C;
    output += blockIdx.x * C;

    MD md_temp, md_val;
    md_val.m = -INFINITY;
    md_val.d = 0.0f;

    for (int i = threadIdx.x; i < C; i += blockDim.x) {
        md_temp.m = input[i];
        md_temp.d = 1.0f;
        md_val = MD_OP()(md_val, md_temp);
    }

    using BlockReduce = cub::BlockReduce<MD, BLOCK_DIM>;
    __shared__ typename BlockReduce::TempStorage tempStorage;
    __shared__ MD shared_md_val;

    md_val = BlockReduce(tempStorage).Reduce(md_val, MD_OP());
    if (threadIdx.x == 0) {
        shared_md_val = md_val;
    }
    __syncthreads();
    md_val = shared_md_val;

    for (int i = threadIdx.x; i < C; i += blockDim.x) {
        output[i] = __expf(input[i] - md_val.m) / md_val.d;
    }
}

void online_softmax(const float* __restrict__ input, float* __restrict__ output, int N, int C, cudaStream_t stream) {
    dim3 blockDim(512);
    dim3 gridDim(N);
    online_softmax_kernel<512><<<gridDim, blockDim, 0, stream>>>(input, output, C);
}
}  // namespace online_softmax1

namespace online_softmax2 {
struct __align__(8) MD {
    float m;
    float d;
};

struct MD_OP {
    __device__ __forceinline__ MD operator()(const MD& a, const MD& b) const {
        MD res;
        res.m = fmaxf(a.m, b.m);
        res.d = a.d * __expf(a.m - res.m) + b.d * __expf(b.m - res.m);
        return res;
    }
};

__device__ __forceinline__ MD shfl_down_val(MD val, int delta) {
    MD res;
    res.m = __shfl_down_sync(0xffffffff, val.m, delta);
    res.d = __shfl_down_sync(0xffffffff, val.d, delta);
    return res;
}

__device__ __forceinline__ MD warp_reduce(MD val) {
    auto op = MD_OP();
#pragma unroll
    for (int delta = 16; delta > 0; delta >>= 1) {
        val = op(val, shfl_down_val(val, delta));
    }
    return val;
}

__device__ __forceinline__ MD block_reduce(MD val) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int warp_num = blockDim.x >> 5;

    __shared__ MD shared[32];
    __shared__ MD shared_val;

    val = warp_reduce(val);

    if (lane == 0) {
        shared[warp] = val;
    }
    __syncthreads();

    if (warp == 0) {
        val = (lane < warp_num) ? shared[lane] : MD{-INFINITY, 0.0f};
        val = warp_reduce(val);
    }

    if (threadIdx.x == 0) {
        shared_val = val;
    }
    __syncthreads();

    val = shared_val;
    return val;
}

template <int BLOCK_DIM>
__global__ void online_softmax_kernel(const float* __restrict__ input, float* __restrict__ output, int C) {
    input += blockIdx.x * C;
    output += blockIdx.x * C;

    MD md_temp, md_val;
    md_val.m = -INFINITY;
    md_val.d = 0.0f;

    for (int i = threadIdx.x; i < C; i += blockDim.x) {
        md_temp.m = input[i];
        md_temp.d = 1.0f;
        md_val = MD_OP()(md_val, md_temp);
    }

    md_val = block_reduce(md_val);

    for (int i = threadIdx.x; i < C; i += blockDim.x) {
        output[i] = __expf(input[i] - md_val.m) / md_val.d;
    }
}

void online_softmax(const float* __restrict__ input, float* __restrict__ output, int N, int C, cudaStream_t stream) {
    dim3 blockDim(512);
    dim3 gridDim(N);
    online_softmax_kernel<512><<<gridDim, blockDim, 0, stream>>>(input, output, C);
}
}  // namespace online_softmax2

struct CompareResult {
    bool passed;
    float max_abs_err;
    int max_err_idx;
};

CompareResult compare_results(const float* ref, const float* test, int N, int C, float eps = 1e-3f) {
    CompareResult result{true, 0.0f, -1};
    for (int i = 0; i < N * C; ++i) {
        float diff = fabsf(ref[i] - test[i]);
        if (diff > result.max_abs_err) {
            result.max_abs_err = diff;
            result.max_err_idx = i;
        }
        if (diff > eps) {
            result.passed = false;
        }
    }
    return result;
}

template <typename LaunchFunc>
float benchmark_kernel(LaunchFunc launch, int warmup_iters, int bench_iters, cudaStream_t stream) {
    for (int i = 0; i < warmup_iters; ++i) {
        launch();
        CHECK_CUDA(cudaGetLastError());
    }
    CHECK_CUDA(cudaStreamSynchronize(stream));

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start, stream));
    for (int i = 0; i < bench_iters; ++i) {
        launch();
    }
    CHECK_CUDA(cudaEventRecord(stop, stream));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaGetLastError());

    float total_ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&total_ms, start, stop));

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return total_ms / bench_iters;
}

int main() {
    constexpr int N = 128;
    constexpr int C = 8192;
    constexpr int THREADS = 512;
    constexpr int WARMUP_ITERS = 20;
    constexpr int BENCH_ITERS = 200;

    const size_t size = static_cast<size_t>(N) * C * sizeof(float);

    float* in = static_cast<float*>(malloc(size));
    float* out_cpu = static_cast<float*>(malloc(size));
    float* out_gpu_1 = static_cast<float*>(malloc(size));
    float* out_gpu_2 = static_cast<float*>(malloc(size));
    float* out_gpu_3 = static_cast<float*>(malloc(size));
    float* out_gpu_4 = static_cast<float*>(malloc(size));

    if (!in || !out_cpu || !out_gpu_1 || !out_gpu_2 || !out_gpu_3 || !out_gpu_4) {
        std::cerr << "Host malloc failed." << std::endl;
        return EXIT_FAILURE;
    }

    std::srand(123);
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < C; ++j) {
            in[i * C + j] = static_cast<float>(std::rand()) / RAND_MAX;
        }
    }

    auto cpu_start = std::chrono::high_resolution_clock::now();
    softmax_forward_cpu(out_cpu, in, N, C);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpu_time = cpu_end - cpu_start;

    float *d_in = nullptr, *d_out_1 = nullptr, *d_out_2 = nullptr, *d_out_3 = nullptr, *d_out_4 = nullptr;
    CHECK_CUDA(cudaMalloc(&d_in, size));
    CHECK_CUDA(cudaMalloc(&d_out_1, size));
    CHECK_CUDA(cudaMalloc(&d_out_2, size));
    CHECK_CUDA(cudaMalloc(&d_out_3, size));
    CHECK_CUDA(cudaMalloc(&d_out_4, size));

    CHECK_CUDA(cudaMemcpy(d_in, in, size, cudaMemcpyHostToDevice));

    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    constexpr int block_num = N;
    constexpr int thread_num = THREADS;

    auto launch_kernel_1 = [&]() {
        softmax::softmax_kernel_1<thread_num><<<block_num, thread_num, 0, stream>>>(d_out_1, d_in, C);
    };

    auto launch_kernel_2 = [&]() {
        softmax::softmax_kernel_2<thread_num><<<block_num, thread_num, 0, stream>>>(d_out_2, d_in, C);
    };

    auto launch_kernel_3 = [&]() {
        online_softmax1::online_softmax_kernel<thread_num><<<block_num, thread_num, 0, stream>>>(d_in, d_out_3, C);
    };

    auto launch_kernel_4 = [&]() {
        online_softmax2::online_softmax_kernel<thread_num><<<block_num, thread_num, 0, stream>>>(d_in, d_out_4, C);
    };

    float kernel1_ms = benchmark_kernel(launch_kernel_1, WARMUP_ITERS, BENCH_ITERS, stream);
    float kernel2_ms = benchmark_kernel(launch_kernel_2, WARMUP_ITERS, BENCH_ITERS, stream);
    float kernel3_ms = benchmark_kernel(launch_kernel_3, WARMUP_ITERS, BENCH_ITERS, stream);
    float kernel4_ms = benchmark_kernel(launch_kernel_4, WARMUP_ITERS, BENCH_ITERS, stream);

    CHECK_CUDA(cudaMemcpy(out_gpu_1, d_out_1, size, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(out_gpu_2, d_out_2, size, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(out_gpu_3, d_out_3, size, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(out_gpu_4, d_out_4, size, cudaMemcpyDeviceToHost));

    CompareResult cmp1 = compare_results(out_cpu, out_gpu_1, N, C);
    CompareResult cmp2 = compare_results(out_cpu, out_gpu_2, N, C);
    CompareResult cmp3 = compare_results(out_cpu, out_gpu_3, N, C);
    CompareResult cmp4 = compare_results(out_cpu, out_gpu_4, N, C);

    float kernel_ms[4] = {kernel1_ms, kernel2_ms, kernel3_ms, kernel4_ms};
    const char* kernel_names[4] = {"Kernel 1", "Kernel 2", "Kernel 3", "Kernel 4"};
    int best_idx = 0;
    for (int i = 1; i < 4; ++i) {
        if (kernel_ms[i] < kernel_ms[best_idx]) {
            best_idx = i;
        }
    }

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "================ Benchmark Result ================\n";
    std::cout << "Shape: N = " << N << ", C = " << C << "\n";
    std::cout << "Threads per block: " << THREADS << "\n";
    std::cout << "Warmup iterations: " << WARMUP_ITERS << "\n";
    std::cout << "Benchmark iterations: " << BENCH_ITERS << "\n\n";

    std::cout << "CPU reference time: " << cpu_time.count() << " ms\n\n";

    std::cout << "[Kernel 1] avg kernel time: " << kernel1_ms << " ms\n";
    std::cout << "[Kernel 1] speedup vs CPU: " << cpu_time.count() / kernel1_ms << "x\n";
    std::cout << "[Kernel 1] correctness: " << (cmp1.passed ? "PASS" : "FAIL") << "\n";
    std::cout << "[Kernel 1] max abs error: " << cmp1.max_abs_err
              << ", max error idx: " << cmp1.max_err_idx << "\n\n";

    std::cout << "[Kernel 2] avg kernel time: " << kernel2_ms << " ms\n";
    std::cout << "[Kernel 2] speedup vs CPU: " << cpu_time.count() / kernel2_ms << "x\n";
    std::cout << "[Kernel 2] correctness: " << (cmp2.passed ? "PASS" : "FAIL") << "\n";
    std::cout << "[Kernel 2] max abs error: " << cmp2.max_abs_err
              << ", max error idx: " << cmp2.max_err_idx << "\n\n";

    std::cout << "[Kernel 3] avg kernel time: " << kernel3_ms << " ms\n";
    std::cout << "[Kernel 3] speedup vs CPU: " << cpu_time.count() / kernel3_ms << "x\n";
    std::cout << "[Kernel 3] correctness: " << (cmp3.passed ? "PASS" : "FAIL") << "\n";
    std::cout << "[Kernel 3] max abs error: " << cmp3.max_abs_err
              << ", max error idx: " << cmp3.max_err_idx << "\n\n";

    std::cout << "[Kernel 4] avg kernel time: " << kernel4_ms << " ms\n";
    std::cout << "[Kernel 4] speedup vs CPU: " << cpu_time.count() / kernel4_ms << "x\n";
    std::cout << "[Kernel 4] correctness: " << (cmp4.passed ? "PASS" : "FAIL") << "\n";
    std::cout << "[Kernel 4] max abs error: " << cmp4.max_abs_err
              << ", max error idx: " << cmp4.max_err_idx << "\n\n";

    std::cout << "Faster kernel: " << kernel_names[best_idx] << "\n";
    std::cout << "==================================================\n";

    CHECK_CUDA(cudaStreamDestroy(stream));
    CHECK_CUDA(cudaFree(d_in));
    CHECK_CUDA(cudaFree(d_out_1));
    CHECK_CUDA(cudaFree(d_out_2));
    CHECK_CUDA(cudaFree(d_out_3));
    CHECK_CUDA(cudaFree(d_out_4));

    free(in);
    free(out_cpu);
    free(out_gpu_1);
    free(out_gpu_2);
    free(out_gpu_3);
    free(out_gpu_4);

    return 0;
}