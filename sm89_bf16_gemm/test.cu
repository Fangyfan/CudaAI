#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace gemm {

void launch_bf16_gemm_kernel(
    const nv_bfloat16* A,
    const nv_bfloat16* B,
    nv_bfloat16* C,
    int M,
    int N,
    int K,
    cudaStream_t stream
);

}  // namespace gemm

#define CHECK_CUDA(call)                                                        \
    do {                                                                        \
        cudaError_t err__ = (call);                                             \
        if (err__ != cudaSuccess) {                                             \
            std::cerr << "CUDA error: " << cudaGetErrorString(err__)            \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl;    \
            std::exit(EXIT_FAILURE);                                            \
        }                                                                       \
    } while (0)

static const char* cublas_status_string(cublasStatus_t status) {
    switch (status) {
        case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE";
        case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH";
        case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED";
        case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR";
        case CUBLAS_STATUS_NOT_SUPPORTED: return "CUBLAS_STATUS_NOT_SUPPORTED";
        case CUBLAS_STATUS_LICENSE_ERROR: return "CUBLAS_STATUS_LICENSE_ERROR";
        default: return "UNKNOWN_CUBLAS_STATUS";
    }
}

#define CHECK_CUBLAS(call)                                                      \
    do {                                                                        \
        cublasStatus_t status__ = (call);                                       \
        if (status__ != CUBLAS_STATUS_SUCCESS) {                                \
            std::cerr << "cuBLAS error: " << cublas_status_string(status__)     \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl;    \
            std::exit(EXIT_FAILURE);                                            \
        }                                                                       \
    } while (0)

struct BenchResult {
    float best_ms;
    float avg_ms;
};

template <typename Fn>
BenchResult benchmark(
    Fn&& fn,
    cudaStream_t stream,
    int warmup_iters,
    int measure_iters,
    int rounds
) {
    for (int i = 0; i < warmup_iters; ++i) {
        fn();
    }

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaStreamSynchronize(stream));

    std::vector<float> round_ms;
    round_ms.reserve(rounds);

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    for (int r = 0; r < rounds; ++r) {
        CHECK_CUDA(cudaEventRecord(start, stream));

        for (int i = 0; i < measure_iters; ++i) {
            fn();
        }

        CHECK_CUDA(cudaEventRecord(stop, stream));
        CHECK_CUDA(cudaEventSynchronize(stop));
        CHECK_CUDA(cudaGetLastError());

        float elapsed_ms = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&elapsed_ms, start, stop));
        round_ms.push_back(elapsed_ms / measure_iters);
    }

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    float best = *std::min_element(round_ms.begin(), round_ms.end());
    float avg = 0.0f;
    for (float x : round_ms) avg += x;
    avg /= static_cast<float>(round_ms.size());

    return {best, avg};
}

static double tflops_from_ms(int M, int N, int K, float ms) {
    const double flops = 2.0 * static_cast<double>(M) * N * K;
    return flops / (static_cast<double>(ms) * 1.0e-3) / 1.0e12;
}

struct ErrorStats {
    double max_abs = 0.0;
    double max_rel = 0.0;
    double mean_abs = 0.0;
    double rmse = 0.0;
    std::size_t bad = 0;
};

ErrorStats compare_result(
    const std::vector<nv_bfloat16>& got,
    const std::vector<nv_bfloat16>& ref,
    double abs_tol,
    double rel_tol
) {
    ErrorStats s;
    const std::size_t size = got.size();

    double sum_abs = 0.0;
    double sum_sq = 0.0;

    for (std::size_t i = 0; i < size; ++i) {
        const float x = __bfloat162float(got[i]);
        const float y = __bfloat162float(ref[i]);

        const double abs_err = std::abs(static_cast<double>(x) - y);
        const double ref_abs = std::abs(static_cast<double>(y));

        double rel_err = 0.0;
        if (ref_abs > 1.0e-2) {
            rel_err = abs_err / ref_abs;
            s.max_rel = std::max(s.max_rel, rel_err);
        }

        s.max_abs = std::max(s.max_abs, abs_err);
        sum_abs += abs_err;
        sum_sq += abs_err * abs_err;

        if (abs_err > abs_tol + rel_tol * ref_abs) {
            ++s.bad;
        }
    }

    s.mean_abs = sum_abs / static_cast<double>(size);
    s.rmse = std::sqrt(sum_sq / static_cast<double>(size));
    return s;
}

void run_cublas_bf16_gemm(
    cublasHandle_t handle,
    const nv_bfloat16* A_row_major,
    const nv_bfloat16* B_col_major,
    nv_bfloat16* C_row_major,
    int M,
    int N,
    int K
) {
    const float alpha = 1.0f;
    const float beta = 0.0f;

    /*
      自定义 kernel:
          C_row[M, N] = A_row[M, K] * B_col[K, N]

      cuBLAS column-major 视角:
          A_row 内存 == column-major 的 A_cm[K, M]，即 A^T
          B_col 内存 == column-major 的 B_cm[K, N]，即 B
          C_row 内存 == column-major 的 C_cm[N, M]，即 C^T

      所以:
          C^T[N, M] = B^T[N, K] * A^T[K, M]

      即 cuBLAS:
          m = N, n = M, k = K
          op(A) = B^T, transa = CUBLAS_OP_T
          op(B) = A^T, transb = CUBLAS_OP_N
    */
    CHECK_CUBLAS(cublasGemmEx(
        handle,
        CUBLAS_OP_T,
        CUBLAS_OP_N,
        N,
        M,
        K,
        &alpha,
        B_col_major,
        CUDA_R_16BF,
        K,
        A_row_major,
        CUDA_R_16BF,
        K,
        &beta,
        C_row_major,
        CUDA_R_16BF,
        N,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP
    ));
}

int main() {
    // constexpr int M = 4096;
    // constexpr int N = 16384;
    // constexpr int K = 4096;

    constexpr int M = 4096;
    constexpr int N = 4096;
    constexpr int K = 4096;
    
    static_assert(M % 128 == 0);
    static_assert(N % 128 == 0);
    static_assert(K % 64 == 0);

    const std::size_t size_A = static_cast<std::size_t>(M) * K;
    const std::size_t size_B = static_cast<std::size_t>(K) * N;
    const std::size_t size_C = static_cast<std::size_t>(M) * N;

    const std::size_t bytes_A = size_A * sizeof(nv_bfloat16);
    const std::size_t bytes_B = size_B * sizeof(nv_bfloat16);
    const std::size_t bytes_C = size_C * sizeof(nv_bfloat16);

    std::cout << "M = " << M << ", N = " << N << ", K = " << K << "\n";
    std::cout << "A bytes: " << bytes_A / 1024.0 / 1024.0 << " MiB\n";
    std::cout << "B bytes: " << bytes_B / 1024.0 / 1024.0 << " MiB\n";
    std::cout << "C bytes: " << bytes_C / 1024.0 / 1024.0 << " MiB\n\n";

    std::vector<nv_bfloat16> h_A(size_A);
    std::vector<nv_bfloat16> h_B(size_B);
    std::vector<nv_bfloat16> h_C_custom(size_C);
    std::vector<nv_bfloat16> h_C_cublas(size_C);

    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);

    // A row-major: A[m * K + k]
    for (std::size_t i = 0; i < size_A; ++i) {
        h_A[i] = __float2bfloat16_rn(dist(rng));
    }

    // B column-major logical [K, N]: B[n * K + k]
    for (std::size_t i = 0; i < size_B; ++i) {
        h_B[i] = __float2bfloat16_rn(dist(rng));
    }

    nv_bfloat16* d_A = nullptr;
    nv_bfloat16* d_B = nullptr;
    nv_bfloat16* d_C_custom = nullptr;
    nv_bfloat16* d_C_cublas = nullptr;

    CHECK_CUDA(cudaMalloc(&d_A, bytes_A));
    CHECK_CUDA(cudaMalloc(&d_B, bytes_B));
    CHECK_CUDA(cudaMalloc(&d_C_custom, bytes_C));
    CHECK_CUDA(cudaMalloc(&d_C_cublas, bytes_C));

    CHECK_CUDA(cudaMemcpy(d_A, h_A.data(), bytes_A, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B.data(), bytes_B, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_C_custom, 0, bytes_C));
    CHECK_CUDA(cudaMemset(d_C_cublas, 0, bytes_C));

    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));
    CHECK_CUBLAS(cublasSetStream(handle, stream));

    constexpr int warmup_iters = 10;
    constexpr int measure_iters = 50;
    constexpr int rounds = 5;

    auto custom_fn = [&]() {
        gemm::launch_bf16_gemm_kernel(d_A, d_B, d_C_custom, M, N, K, stream);
    };

    auto cublas_fn = [&]() {
        run_cublas_bf16_gemm(handle, d_A, d_B, d_C_cublas, M, N, K);
    };

    // 先各跑一次，提前暴露 launch/config 错误。
    custom_fn();
    cublas_fn();
    CHECK_CUDA(cudaStreamSynchronize(stream));
    CHECK_CUDA(cudaGetLastError());

    BenchResult custom_ms = benchmark(custom_fn, stream, warmup_iters, measure_iters, rounds);
    BenchResult cublas_ms = benchmark(cublas_fn, stream, warmup_iters, measure_iters, rounds);

    CHECK_CUDA(cudaMemcpy(h_C_custom.data(), d_C_custom, bytes_C, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_C_cublas.data(), d_C_cublas, bytes_C, cudaMemcpyDeviceToHost));

    // BF16 输出比较：阈值不要设得像 FP32 那么严。
    // const double abs_tol = 2.0;
    // const double rel_tol = 5.0e-2;
    const double abs_tol = 0.25;
    const double rel_tol = 2.0e-2;
    ErrorStats err = compare_result(h_C_custom, h_C_cublas, abs_tol, rel_tol);

    std::cout << std::fixed << std::setprecision(4);

    std::cout << "Custom kernel:\n";
    std::cout << "  best ms  = " << custom_ms.best_ms << "\n";
    std::cout << "  avg  ms  = " << custom_ms.avg_ms << "\n";
    std::cout << "  best TFLOPS = " << tflops_from_ms(M, N, K, custom_ms.best_ms) << "\n";
    std::cout << "  avg  TFLOPS = " << tflops_from_ms(M, N, K, custom_ms.avg_ms) << "\n\n";

    std::cout << "cuBLAS BF16 GEMM:\n";
    std::cout << "  best ms  = " << cublas_ms.best_ms << "\n";
    std::cout << "  avg  ms  = " << cublas_ms.avg_ms << "\n";
    std::cout << "  best TFLOPS = " << tflops_from_ms(M, N, K, cublas_ms.best_ms) << "\n";
    std::cout << "  avg  TFLOPS = " << tflops_from_ms(M, N, K, cublas_ms.avg_ms) << "\n\n";

    std::cout << "Accuracy vs cuBLAS:\n";
    std::cout << "  max_abs  = " << err.max_abs << "\n";
    std::cout << "  max_rel  = " << err.max_rel << "\n";
    std::cout << "  mean_abs = " << err.mean_abs << "\n";
    std::cout << "  rmse     = " << err.rmse << "\n";
    std::cout << "  bad      = " << err.bad << " / " << size_C
              << "  with abs_tol=" << abs_tol
              << ", rel_tol=" << rel_tol << "\n\n";

    if (err.bad == 0) {
        std::cout << "PASS\n";
    } else {
        std::cout << "FAIL or needs investigation\n";
    }

    CHECK_CUBLAS(cublasDestroy(handle));
    CHECK_CUDA(cudaStreamDestroy(stream));

    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C_custom));
    CHECK_CUDA(cudaFree(d_C_cublas));

    return 0;
}