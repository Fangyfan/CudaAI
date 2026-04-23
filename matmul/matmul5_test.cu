#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <iomanip>
#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>

void checkCudaError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << msg << " CUDA ERROR: " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

void checkCublasError(cublasStatus_t status, const char* msg) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cerr << msg << " CUBLAS ERROR: " << status << std::endl;
        exit(EXIT_FAILURE);
    }
}

constexpr int WARP_SIZE = 32;
#define FLOAT4(p) (*reinterpret_cast<float4*>(&(p)))
#define OFFSET(y, x, ld) ((y) * (ld) + (x))

template<int BLOCK_DIM, int BM, int BN, int BK, int WM, int WN, int WM_ITER, int WN_ITER, int TM, int TN>
__global__ void __launch_bounds__(BLOCK_DIM) mysgemm_v6(int M, int N, int K, float alpha, float* A, float* B, float beta, float* C) {
    int bx = blockIdx.x * BN;
    int by = blockIdx.y * BM;

    int warp_id = threadIdx.x / WARP_SIZE;
    int wx = warp_id % (BN / WN) * WN;
    int wy = warp_id / (BN / WN) * WM;

    constexpr int WM_SUB = WM / WM_ITER;
    constexpr int WN_SUB = WN / WN_ITER;

    int lane_id = threadIdx.x % WARP_SIZE;
    int tx = lane_id % (WN_SUB / TN) * TN;
    int ty = lane_id / (WN_SUB / TN) * TM;

    A += by * K;
    B += bx;
    C += (by + wy) * N + (bx + wx);

    __shared__ float As[BK * BM];
    __shared__ float Bs[BK * BN];

    int a_tile_x = threadIdx.x % (BK / 4) * 4;
    int a_tile_y = threadIdx.x / (BK / 4);
    int b_tile_x = threadIdx.x % (BN / 4) * 4;
    int b_tile_y = threadIdx.x / (BN / 4);

    constexpr int a_tile_stride = (BLOCK_DIM * 4) / BK;
    constexpr int b_tile_stride = (BLOCK_DIM * 4) / BN;

    float temp[TM * WM_ITER * TN * WN_ITER] = { 0.0f };
    float a_vec[TM * WM_ITER];
    float b_vec[TN * WN_ITER];

    for (int k = 0; k < K; k += BK) {
        for (int i = 0; i < BM; i += a_tile_stride) {
            float4 a4 = FLOAT4(A[OFFSET(a_tile_y + i, a_tile_x, K)]);
            As[OFFSET(a_tile_x, a_tile_y + i, BM)] = a4.x;
            As[OFFSET(a_tile_x + 1, a_tile_y + i, BM)] = a4.y;
            As[OFFSET(a_tile_x + 2, a_tile_y + i, BM)] = a4.z;
            As[OFFSET(a_tile_x + 3, a_tile_y + i, BM)] = a4.w;
        }
        for (int i = 0; i < BK; i += b_tile_stride) {
            FLOAT4(Bs[OFFSET(b_tile_y + i, b_tile_x, BN)])
            = FLOAT4(B[OFFSET(b_tile_y + i, b_tile_x, N)]);
        }
        __syncthreads();

        A += BK;
        B += BK * N;

        for (int x = 0; x < BK; ++x) {
            for (int sub_y = 0; sub_y < WM_ITER; ++sub_y) {
                for (int i = 0; i < TM; i += 4) {
                    FLOAT4(a_vec[sub_y * TM + i]) = FLOAT4(As[OFFSET(x, wy + sub_y * WM_SUB + ty + i, BM)]);
                }
            }
            for (int sub_x = 0; sub_x < WN_ITER; ++sub_x) {
                for (int j = 0; j < TN; j += 4) {
                    FLOAT4(b_vec[sub_x * TN + j]) = FLOAT4(Bs[OFFSET(x, wx + sub_x * WN_SUB + tx + j, BN)]);
                }
            }
            for (int sub_y = 0; sub_y < WM_ITER; ++sub_y) {
                for (int sub_x = 0; sub_x < WN_ITER; ++sub_x) {
                    for (int i = 0; i < TM; ++i) {
                        for (int j = 0; j < TN; ++j) {
                            temp[OFFSET(sub_y * TM + i, sub_x * TN + j, TN * WN_ITER)]
                            += a_vec[sub_y * TM + i] * b_vec[sub_x * TN + j];
                        }
                    }
                }
            }
        }
        __syncthreads();
    }

    for (int sub_y = 0; sub_y < WM_ITER; ++sub_y) {
        for (int sub_x = 0; sub_x < WN_ITER; ++sub_x) {
            float* C_sub = C + OFFSET(sub_y * WM_SUB, sub_x * WN_SUB, N);
            for (int i = 0; i < TM; ++i) {
                for (int j = 0; j < TN; j += 4) {
                    int idx = OFFSET(sub_y * TM + i, sub_x * TN + j, TN * WN_ITER);
                    float4 c4 = FLOAT4(C_sub[OFFSET(ty + i, tx + j, N)]);
                    FLOAT4(C_sub[OFFSET(ty + i, tx + j, N)]) = make_float4(
                        alpha * temp[idx] + beta * c4.x,
                        alpha * temp[idx + 1] + beta * c4.y,
                        alpha * temp[idx + 2] + beta * c4.z,
                        alpha * temp[idx + 3] + beta * c4.w
                    );
                }
            }
        }
    }
}

struct CompareResult {
    int error_count;
    float max_rel_err;
};

struct OperatorBandwidthResult {
    double flops;
    double gmem_bytes_requested;
    double smem_bytes_requested;
    double gmem_effective_GBs;
    double smem_effective_GBs;
    double gmem_ai;
    double smem_ai;
};

template<int BLOCK_DIM, int BM, int BN, int BK, int WM, int WN, int WM_ITER, int WN_ITER, int TM, int TN>
OperatorBandwidthResult estimate_mysgemm_v6_bandwidth(int M, int N, int K, float kernel_time_ms) {
    OperatorBandwidthResult r{};

    const int grid_x = N / BN;
    const int grid_y = M / BM;
    const int block_num = grid_x * grid_y;
    const int k_tiles = K / BK;

    const double flops_per_launch = 2.0 * static_cast<double>(M) * N * K;

    const double gmem_ab_floats_per_block = static_cast<double>(k_tiles) * (BM * BK + BK * BN);
    const double gmem_c_floats_per_block = 2.0 * BM * BN;
    const double gmem_floats_per_block = gmem_ab_floats_per_block + gmem_c_floats_per_block;
    const double gmem_bytes_per_launch =
        static_cast<double>(block_num) * gmem_floats_per_block * sizeof(float);

    const double smem_write_floats_per_block = static_cast<double>(k_tiles) * (BM * BK + BK * BN);
    const double smem_read_floats_per_block =
        static_cast<double>(k_tiles) * BK * BLOCK_DIM * (WM_ITER * TM + WN_ITER * TN);
    const double smem_floats_per_block = smem_write_floats_per_block + smem_read_floats_per_block;
    const double smem_bytes_per_launch =
        static_cast<double>(block_num) * smem_floats_per_block * sizeof(float);

    const double kernel_time_s = kernel_time_ms * 1e-3;

    r.flops = flops_per_launch;
    r.gmem_bytes_requested = gmem_bytes_per_launch;
    r.smem_bytes_requested = smem_bytes_per_launch;
    r.gmem_effective_GBs = gmem_bytes_per_launch / 1e9 / kernel_time_s;
    r.smem_effective_GBs = smem_bytes_per_launch / 1e9 / kernel_time_s;
    r.gmem_ai = flops_per_launch / gmem_bytes_per_launch;
    r.smem_ai = flops_per_launch / smem_bytes_per_launch;
    return r;
}

CompareResult compare_results(const float* ref, const float* test, int num_elem, float rel_eps = 1e-4f) {
    CompareResult result{0, 0.0f};
    for (int i = 0; i < num_elem; ++i) {
        float ref_val = ref[i];
        float test_val = test[i];
        float diff = fabsf(ref_val - test_val);
        float rel_err = diff / (fabsf(ref_val) + 1e-7f);
        if (rel_err > result.max_rel_err) {
            result.max_rel_err = rel_err;
        }
        if (rel_err > rel_eps) {
            result.error_count++;
            if (result.error_count >= 10) {
                break;
            }
        }
    }
    return result;
}

double get_peak_gmem_bandwidth_GBs() {
    cudaDeviceProp prop;
    checkCudaError(cudaGetDeviceProperties(&prop, 0), "cudaGetDeviceProperties failed");
    return static_cast<double>(prop.memoryClockRate) * 1e3 *
           (static_cast<double>(prop.memoryBusWidth) / 8.0) * 2.0 / 1e9;
}

double get_peak_smem_bandwidth_GBs() {
    cudaDeviceProp prop;
    checkCudaError(cudaGetDeviceProperties(&prop, 0), "cudaGetDeviceProperties failed");
    double sm_clock_hz = static_cast<double>(prop.clockRate) * 1e3;
    double bytes_per_sm_per_cycle = 32.0 * 4.0;
    return bytes_per_sm_per_cycle * prop.multiProcessorCount * sm_clock_hz / 1e9;
}

std::vector<int> generate() {
    std::vector<int> sizes;
    for (int i = 256; i <= 8192; i += 256) {
        sizes.push_back(i);
    }
    return sizes;
}

int main() {
    std::vector<int> sizes = {128, 256, 512, 1024, 2048, 4096};

    std::ofstream ofs("sgemm_benchmark_v6_operator_bandwidth.csv");
    ofs << "Size,";
    ofs << "CUBLAS_GFLOPS,";
    ofs << "MySGEMM_GFLOPS,";
    ofs << "UnMatched,";
    ofs << "Ratio,";
    ofs << "MySGEMM_GMEM_GBps,";
    ofs << "MySGEMM_SMEM_GBps,";
    ofs << "MySGEMM_GMEM_AI_FlopPerByte,";
    ofs << "MySGEMM_SMEM_AI_FlopPerByte,";
    ofs << "Peak_GMEM_GBps,";
    ofs << "Peak_SMEM_GBps,";
    ofs << "GMEM_Util,";
    ofs << "SMEM_Util"
        << std::endl;

    const double peak_gmem_GBs = get_peak_gmem_bandwidth_GBs();
    const double peak_smem_GBs = get_peak_smem_bandwidth_GBs();

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Peak Global Memory Bandwidth (theoretical): " << peak_gmem_GBs << " GB/s" << std::endl;
    std::cout << "Peak Shared Memory Bandwidth (theoretical): " << peak_smem_GBs << " GB/s" << std::endl;
    std::cout << std::endl;

    for (int N : sizes) {
        std::cout << "Testing size: " << N << std::endl;
        size_t size = static_cast<size_t>(N) * N * sizeof(float);
        float* A = (float*)malloc(size);
        float* B = (float*)malloc(size);
        float* C_cublas = (float*)malloc(size);
        float* C_v6 = (float*)malloc(size);

        float* d_A = nullptr;
        float* d_B = nullptr;
        float* d_C_v6 = nullptr;
        checkCudaError(cudaMalloc(reinterpret_cast<void**>(&d_A), size), "cudaMalloc d_A failed");
        checkCudaError(cudaMalloc(reinterpret_cast<void**>(&d_B), size), "cudaMalloc d_B failed");
        checkCudaError(cudaMalloc(reinterpret_cast<void**>(&d_C_v6), size), "cudaMalloc d_C_v6 failed");

        bool out_of_memory = false;
        float alpha = 1.0f;
        float beta = 0.0f;

        for (int i = 0; i < N * N; ++i) {
            A[i] = (float)rand() / RAND_MAX;
            B[i] = (float)rand() / RAND_MAX;
        }

        try {
            checkCudaError(cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice), "cudaMemcpy A to device failed");
            checkCudaError(cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice), "cudaMemcpy B to device failed");

            cublasHandle_t handle;
            checkCublasError(cublasCreate(&handle), "cublasCreate handle failed");

            cudaEvent_t start, end;
            checkCudaError(cudaEventCreate(&start), "cudaEventCreate(start) failed");
            checkCudaError(cudaEventCreate(&end), "cudaEventCreate(end) failed");

            int warmup_times = 10;
            for (int i = 0; i < warmup_times; ++i) {
                checkCublasError(
                    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                N, N, N,
                                &alpha, d_B, N,
                                d_A, N,
                                &beta, d_C_v6, N),
                    "warm up cublasSgemm failed"
                );
            }
            checkCudaError(cudaDeviceSynchronize(), "cudaDeviceSynchronize after cublas warmup failed");

            int repeat_times = 50;
            float cublas_time_ms = 0.0f;
            checkCudaError(cudaEventRecord(start), "cublas cudaEventRecord(start) failed");
            for (int i = 0; i < repeat_times; ++i) {
                checkCublasError(
                    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                N, N, N,
                                &alpha, d_B, N,
                                d_A, N,
                                &beta, d_C_v6, N),
                    "cublasSgemm failed"
                );
            }
            checkCudaError(cudaEventRecord(end), "cublas cudaEventRecord(end) failed");
            checkCudaError(cudaEventSynchronize(end), "cublas cudaEventSynchronize(end) failed");
            checkCudaError(cudaEventElapsedTime(&cublas_time_ms, start, end), "cublas cudaEventElapsedTime failed");

            checkCudaError(cudaMemcpy(C_cublas, d_C_v6, size, cudaMemcpyDeviceToHost), "cudaMemcpy C_cublas failed");
            checkCudaError(cudaMemset(d_C_v6, 0, size), "cudaMemset d_C_v6 failed");

            constexpr int BLOCK_DIM = 128;
            constexpr int BN = 128;
            constexpr int BM = 128;
            constexpr int BK = 16;
            constexpr int WM = 64;
            constexpr int WN = 64;
            constexpr int WN_ITER = 2;
            constexpr int TM = 8;
            constexpr int TN = 4;
            static_assert((WM * WN) % (WARP_SIZE * TM * TN * WN_ITER) == 0);
            constexpr int WM_ITER = (WM * WN) / (WARP_SIZE * TM * TN * WN_ITER);

            static_assert(BN % WN == 0);
            static_assert(WM % WM_ITER == 0);
            static_assert(WN % WN_ITER == 0);
            static_assert((WN / WN_ITER) % TN == 0);
            static_assert(BN % 4 == 0);
            static_assert(BK % 4 == 0);
            static_assert((4 * BLOCK_DIM) % BK == 0);
            static_assert((4 * BLOCK_DIM) % BN == 0);

            if ((N % BM) != 0 || (N % BN) != 0 || (N % BK) != 0) {
                std::cerr << "Current kernel requires N divisible by BM/BN/BK. Skip size: " << N << std::endl;
                out_of_memory = true;
            }

            dim3 blockDim(BLOCK_DIM);
            dim3 gridDim(N / BN, N / BM);

            if (!out_of_memory) {
                for (int i = 0; i < warmup_times; ++i) {
                    mysgemm_v6<BLOCK_DIM, BM, BN, BK, WM, WN, WM_ITER, WN_ITER, TM, TN><<<gridDim, blockDim>>>(
                        N, N, N, alpha, d_A, d_B, beta, d_C_v6
                    );
                }
                checkCudaError(cudaDeviceSynchronize(), "cudaDeviceSynchronize after mysgemm warmup failed");

                float v6_time_ms = 0.0f;
                checkCudaError(cudaEventRecord(start), "v6 cudaEventRecord(start) failed");
                for (int i = 0; i < repeat_times; ++i) {
                    mysgemm_v6<BLOCK_DIM, BM, BN, BK, WM, WN, WM_ITER, WN_ITER, TM, TN><<<gridDim, blockDim>>>(
                        N, N, N, alpha, d_A, d_B, beta, d_C_v6
                    );
                }
                checkCudaError(cudaEventRecord(end), "v6 cudaEventRecord(end) failed");
                checkCudaError(cudaEventSynchronize(end), "v6 cudaEventSynchronize(end) failed");
                checkCudaError(cudaEventElapsedTime(&v6_time_ms, start, end), "v6 cudaEventElapsedTime failed");

                checkCudaError(cudaMemcpy(C_v6, d_C_v6, size, cudaMemcpyDeviceToHost), "cudaMemcpy C_v6 failed");

                CompareResult cmp = compare_results(C_cublas, C_v6, N * N);

                float cublas_gflops = repeat_times * 2.0f * N * N * N / (cublas_time_ms * 1e6f);
                float v6_gflops = repeat_times * 2.0f * N * N * N / (v6_time_ms * 1e6f);

                OperatorBandwidthResult bw = estimate_mysgemm_v6_bandwidth<
                    BLOCK_DIM, BM, BN, BK, WM, WN, WM_ITER, WN_ITER, TM, TN
                >(N, N, N, v6_time_ms / repeat_times);

                std::cout << "  cuBLAS GFLOPS                : " << cublas_gflops << std::endl;
                std::cout << "  mysgemm_v6 GFLOPS            : " << v6_gflops << std::endl;
                std::cout << "  mysgemm_v6 GMEM GB/s         : " << bw.gmem_effective_GBs << std::endl;
                std::cout << "  mysgemm_v6 SMEM GB/s         : " << bw.smem_effective_GBs << std::endl;
                std::cout << "  mysgemm_v6 GMEM AI           : " << bw.gmem_ai << " flop/byte" << std::endl;
                std::cout << "  mysgemm_v6 SMEM AI           : " << bw.smem_ai << " flop/byte" << std::endl;
                std::cout << "  mysgemm_v6 GMEM Utilization  : " << 100.0 * bw.gmem_effective_GBs / peak_gmem_GBs << "%" << std::endl;
                std::cout << "  mysgemm_v6 SMEM Utilization  : " << 100.0 * bw.smem_effective_GBs / peak_smem_GBs << "%" << std::endl;
                std::cout << "  max relative error           : " << cmp.max_rel_err << std::endl;
                std::cout << "  mismatch count (<=10 stop)   : " << cmp.error_count << std::endl;

                ofs << N << ",";
                ofs << cublas_gflops << ",";
                ofs << v6_gflops << ",";
                ofs << cmp.error_count << ",";
                ofs << std::fixed << std::setprecision(2) << (100.0 * v6_gflops / cublas_gflops) << "%" << ",";
                ofs << std::setprecision(6) << bw.gmem_effective_GBs << ",";
                ofs << bw.smem_effective_GBs << ",";
                ofs << bw.gmem_ai << ",";
                ofs << bw.smem_ai << ",";
                ofs << peak_gmem_GBs << ",";
                ofs << peak_smem_GBs << ",";
                ofs << (100.0 * bw.gmem_effective_GBs / peak_gmem_GBs) << "%" << ",";
                ofs << (100.0 * bw.smem_effective_GBs / peak_smem_GBs) << "%" << std::endl;
            }

            cublasDestroy(handle);
            cudaEventDestroy(start);
            cudaEventDestroy(end);
            cudaFree(d_A);
            cudaFree(d_B);
            cudaFree(d_C_v6);
            free(A);
            free(B);
            free(C_cublas);
            free(C_v6);
        } catch (...) {
            std::cerr << "Out of memory or error during testing size: " << N << std::endl;
            out_of_memory = true;
        }

        if (out_of_memory) {
            ofs << N << ",OOM,OOM,OOM,OOM,OOM,OOM,OOM,OOM,OOM,OOM,OOM,OOM" << std::endl;
        }

        std::cout << "Finished size: " << N << std::endl << std::endl;
    }

    ofs.close();
    std::cout << "Benchmark completed. Results saved to 'sgemm_benchmark_v6_operator_bandwidth.csv'" << std::endl;
    std::cout << "Note: GMEM/SMEM bandwidth reported by this code are kernel requested/effective bandwidths." << std::endl;
    std::cout << "For transaction-level actual DRAM/shared-memory traffic, use Nsight Compute counters." << std::endl;

    return 0;
}
