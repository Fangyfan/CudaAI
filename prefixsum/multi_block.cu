#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <cstdlib>

#define CHECK_CUDA(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err)            \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)

__global__ void kogge_stone_scan1(float* in, float* out, float* sum, int n) {
    extern __shared__ float T[];

    int i = threadIdx.x;
    int offset = blockIdx.x * blockDim.x;
    T[i] = (offset + i < n) ? in[offset + i] : 0.0f;
    __syncthreads();

    float temp;
    for (int stride = 1; stride < blockDim.x; stride <<= 1) {
        if (i >= stride) {
            temp = T[i - stride] + T[i];
        }
        __syncthreads();
        if (i >= stride) {
            T[i] = temp;
        }
        __syncthreads();
    }

    if (offset + i < n) out[offset + i] = T[i];
    if (i == blockDim.x - 1) sum[blockIdx.x] = T[blockDim.x - 1];
}

__global__ void kogge_stone_scan2(float* in, float* out, int n) {
    extern __shared__ float T[];

    int i = threadIdx.x;
    T[i] = i < n ? in[i] : 0.0f;
    __syncthreads();

    float sum;
    for (int stride = 1; stride < n; stride <<= 1) {
        if (i >= stride) {
            sum = T[i - stride] + T[i];
        }
        __syncthreads();
        if (i >= stride) {
            T[i] = sum;
        }
        __syncthreads();
    }

    if (i < n) out[i] = T[i];
}

__global__ void kogge_stone_scan11(float* in, float* out, float* sum, int n) {
    extern __shared__ float T[];
    float* source = T;
    float* destination = T + blockDim.x;

    int i = threadIdx.x;
    int offset = blockIdx.x * blockDim.x;
    source[i] = offset + i < n ? in[offset + i] : 0.0f;
    
    for (int stride = 1; stride < blockDim.x; stride <<= 1) {
        __syncthreads();
        destination[i] = source[i];
        if (i >= stride) {
            destination[i] += source[i - stride];
        }
        float* temp = source;
        source = destination;
        destination = temp;
    }

    if (offset + i < n) out[offset + i] = source[i];
    if (i == blockDim.x - 1) sum[blockIdx.x] = source[blockDim.x - 1];
}

__global__ void kogge_stone_scan22(float* in, float* out, int n) {
    extern __shared__ float T[];
    float* source = T;
    float* destination = T + n;

    int i = threadIdx.x;
    source[i] = i < n ? in[i] : 0.0f;
    
    for (int stride = 1; stride < n; stride <<= 1) {
        __syncthreads();
        destination[i] = source[i];
        if (i >= stride) {
            destination[i] += source[i - stride];
        }
        float* temp = source;
        source = destination;
        destination = temp;
    }

    if (i < n) out[i] = source[i];
}

__global__ void kogge_stone_add_sums(float* out, float* sums, int n) {
    if (blockIdx.x == 0) {
        return;
    }

    int i = threadIdx.x;
    int offset = blockIdx.x * blockDim.x;
    float add_val = sums[blockIdx.x - 1];

    if (offset + i < n) out[offset + i] += add_val;
}

__global__ void work_efficient_scan1(float* in, float* out, float* sum) {
    extern __shared__ float T[];

    int i = threadIdx.x;
    int m = 2 * blockDim.x;
    int offset = blockIdx.x * m;
    T[2 * i] = in[offset + 2 * i];
    T[2 * i + 1] = in[offset + 2 * i + 1];

    for (int stride = 1; stride < m; stride <<= 1) {
        __syncthreads();
        int idx = (i + 1) * (2 * stride) - 1;
        if (stride <= idx && idx < m) {
            T[idx] += T[idx - stride];
        }
    }

    for (int stride = (blockDim.x >> 1); stride > 0; stride >>= 1) {
        __syncthreads();
        int idx = (i + 1) * (2 * stride) - 1;
        if (idx + stride < m) {
            T[idx + stride] += T[idx];
        }
    }
    __syncthreads();

    out[offset + 2 * i] = T[2 * i];
    out[offset + 2 * i + 1] = T[2 * i + 1];
    if (i == blockDim.x - 1) sum[blockIdx.x] = T[m - 1];
}

__global__ void work_efficient_scan2(float* in, float* out, int n) {
    extern __shared__ float T[];

    int i = threadIdx.x;
    T[2 * i] = in[2 * i];
    T[2 * i + 1] = in[2 * i + 1];

    for (int stride = 1; stride < n; stride <<= 1) {
        __syncthreads();
        int idx = (i + 1) * (2 * stride) - 1;
        if (stride <= idx && idx < n) {
            T[idx] += T[idx - stride];
        }
    }

    for (int stride = (blockDim.x >> 1); stride > 0; stride >>= 1) {
        __syncthreads();
        int idx = (i + 1) * (2 * stride) - 1;
        if (idx + stride < n) {
            T[idx + stride] += T[idx];
        }
    }
    __syncthreads();

    out[2 * i] = T[2 * i];
    out[2 * i + 1] = T[2 * i + 1];
}

__global__ void work_efficient_add_sums(float* out, float* sums) {
    if (blockIdx.x == 0) {
        return;
    }

    int i = threadIdx.x;
    int offset = blockIdx.x * (2 * blockDim.x);
    float add_val = sums[blockIdx.x - 1];

    out[offset + 2 * i] += add_val;
    out[offset + 2 * i + 1] += add_val;
}

struct VerifyResult {
    bool passed;
    float max_abs_err;
    float max_rel_err;
    int max_err_idx;
};

struct BenchmarkResult {
    const char* name;
    float avg_ms;
    VerifyResult verify;
};

static std::vector<float> make_reference(const std::vector<float>& h_in) {
    std::vector<float> h_ref(h_in.size());
    for (size_t i = 0; i < h_in.size(); ++i) {
        h_ref[i] = h_in[i];
        if (i) h_ref[i] += h_ref[i - 1];
    }
    return h_ref;
}

static VerifyResult verify_output(const std::vector<float>& h_ref,
                                  const std::vector<float>& h_out,
                                  float rel_tol = 1e-4f,
                                  float abs_tol = 1e-6f) {
    VerifyResult result{true, 0.0f, 0.0f, -1};
    for (size_t i = 0; i < h_ref.size(); ++i) {
        float ref = h_ref[i];
        float val = h_out[i];
        float abs_err = std::fabs(val - ref);
        float rel_err = abs_err / (std::fabs(ref) + 1e-7f);
        if (abs_err > result.max_abs_err || rel_err > result.max_rel_err) {
            result.max_abs_err = abs_err;
            result.max_rel_err = rel_err;
            result.max_err_idx = static_cast<int>(i);
        }
        if (abs_err > abs_tol && rel_err > rel_tol) {
            result.passed = false;
        }
    }
    return result;
}

template <typename LaunchFunc>
float benchmark_cuda(LaunchFunc launch, int warmup_iters, int bench_iters) {
    for (int i = 0; i < warmup_iters; ++i) {
        launch();
        CHECK_CUDA(cudaGetLastError());
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < bench_iters; ++i) {
        launch();
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaGetLastError());

    float total_ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&total_ms, start, stop));

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return total_ms / bench_iters;
}

void print_result(const BenchmarkResult& result) {
    std::cout << std::fixed << std::setprecision(6);
    std::cout << std::left << std::setw(22) << result.name
              << " avg_ms=" << std::setw(8) << result.avg_ms
              << " res=" << std::setw(4) << (result.verify.passed ? "PASS" : "FAIL")
              << " max_abs_err=" << std::setw(8) << result.verify.max_abs_err
              << " max_rel_err=" << std::setw(8) << result.verify.max_rel_err
              << std::endl;
}

BenchmarkResult benchmark1(const std::vector<float>& h_in, const std::vector<float>& h_ref) {
    constexpr int n = 1024 * 1024;
    constexpr int thread_num = 1024;
    constexpr int block_num = n / thread_num;
    constexpr int warmup_iters = 10;
    constexpr int bench_iters = 50;

    std::vector<float> h_out(n);

    float* d_in = nullptr;
    float* d_sum = nullptr;
    float* d_out = nullptr;
    CHECK_CUDA(cudaMalloc(&d_in, n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_sum, block_num * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_out, n * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_in, h_in.data(), n * sizeof(float), cudaMemcpyHostToDevice));

    auto launch = [&]() {
        kogge_stone_scan1<<<block_num, thread_num, thread_num * sizeof(float)>>>(d_in, d_out, d_sum, n);
        kogge_stone_scan2<<<1, block_num, block_num * sizeof(float)>>>(d_sum, d_sum, block_num);
        kogge_stone_add_sums<<<block_num, thread_num>>>(d_out, d_sum, n);
    };

    float avg_ms = benchmark_cuda(launch, warmup_iters, bench_iters);

    CHECK_CUDA(cudaMemcpy(h_out.data(), d_out, n * sizeof(float), cudaMemcpyDeviceToHost));
    VerifyResult verify = verify_output(h_ref, h_out);

    CHECK_CUDA(cudaFree(d_in));
    CHECK_CUDA(cudaFree(d_out));
    CHECK_CUDA(cudaFree(d_sum));

    return {"Kogge-Stone (in-place)", avg_ms, verify};
}

BenchmarkResult benchmark2(const std::vector<float>& h_in, const std::vector<float>& h_ref) {
    constexpr int n = 1024 * 1024;
    constexpr int thread_num = 1024;
    constexpr int block_num = n / thread_num;
    constexpr int warmup_iters = 10;
    constexpr int bench_iters = 50;

    std::vector<float> h_out(n);

    float* d_in = nullptr;
    float* d_sum = nullptr;
    float* d_out = nullptr;
    CHECK_CUDA(cudaMalloc(&d_in, n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_sum, block_num * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_out, n * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_in, h_in.data(), n * sizeof(float), cudaMemcpyHostToDevice));

    auto launch = [&]() {
        kogge_stone_scan11<<<block_num, thread_num, 2 * thread_num * sizeof(float)>>>(d_in, d_out, d_sum, n);
        kogge_stone_scan22<<<1, block_num, 2 * block_num * sizeof(float)>>>(d_sum, d_sum, block_num);
        kogge_stone_add_sums<<<block_num, thread_num>>>(d_out, d_sum, n);
    };

    float avg_ms = benchmark_cuda(launch, warmup_iters, bench_iters);

    CHECK_CUDA(cudaMemcpy(h_out.data(), d_out, n * sizeof(float), cudaMemcpyDeviceToHost));
    VerifyResult verify = verify_output(h_ref, h_out);

    CHECK_CUDA(cudaFree(d_in));
    CHECK_CUDA(cudaFree(d_out));
    CHECK_CUDA(cudaFree(d_sum));

    return {"Kogge-Stone (2-buffer)", avg_ms, verify};
}

BenchmarkResult benchmark3(const std::vector<float>& h_in, const std::vector<float>& h_ref) {
    constexpr int n = 1024 * 1024;
    constexpr int thread_num = 512;
    constexpr int element_num = 2 * thread_num;
    constexpr int block_num = n / element_num;
    constexpr int warmup_iters = 10;
    constexpr int bench_iters = 50;

    std::vector<float> h_out(n);

    float* d_in = nullptr;
    float* d_sum = nullptr;
    float* d_out = nullptr;
    CHECK_CUDA(cudaMalloc(&d_in, n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_sum, block_num * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_out, n * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_in, h_in.data(), n * sizeof(float), cudaMemcpyHostToDevice));

    auto launch = [&]() {
        work_efficient_scan1<<<block_num, thread_num, element_num * sizeof(float)>>>(d_in, d_out, d_sum);
        work_efficient_scan2<<<1, block_num / 2, block_num * sizeof(float)>>>(d_sum, d_sum, block_num);
        work_efficient_add_sums<<<block_num, thread_num>>>(d_out, d_sum);
    };

    float avg_ms = benchmark_cuda(launch, warmup_iters, bench_iters);

    CHECK_CUDA(cudaMemcpy(h_out.data(), d_out, n * sizeof(float), cudaMemcpyDeviceToHost));
    VerifyResult verify = verify_output(h_ref, h_out);

    CHECK_CUDA(cudaFree(d_in));
    CHECK_CUDA(cudaFree(d_out));
    CHECK_CUDA(cudaFree(d_sum));

    return {"Blelloch-Scan", avg_ms, verify};
}

int main() {
    constexpr int n = 1024 * 1024;

    std::vector<float> h_in(n);
    std::srand(123);
    for (int i = 0; i < n; ++i) {
        h_in[i] = static_cast<float>(std::rand()) / RAND_MAX;
    }

    std::vector<float> h_ref = make_reference(h_in);

    BenchmarkResult r1 = benchmark1(h_in, h_ref);
    BenchmarkResult r2 = benchmark2(h_in, h_ref);
    BenchmarkResult r3 = benchmark3(h_in, h_ref);

    std::cout << "===============================================================\n";
    std::cout << "Inclusive Scan Benchmark (end-to-end pipeline timing)\n";
    std::cout << "n = " << n << "\n";
    std::cout << "===============================================================\n";
    print_result(r1);
    print_result(r2);
    print_result(r3);
    std::cout << "===============================================================\n";

    return 0;
}
