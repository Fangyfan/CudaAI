#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cmath>    // for fabsf
#include <fstream>  // for CSV output
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

#define BLOCK_SIZE 32

// C[M, N] = A[M, K] × B[K, N]
__global__ void mysgemm_v2(int M, int N, int K, float alpha, float* A, float* B, float beta, float* C) {
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    constexpr int BM = BLOCK_SIZE;
    constexpr int BN = BLOCK_SIZE;
    constexpr int BK = BLOCK_SIZE;

    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    A = &A[by * BM * K];
    B = &B[bx * BN];
    C = &C[by * BM * N + bx * BN];

    float sum = 0.0f;
    for (int k = 0; k < K; k += BK) {
        As[ty][tx] = A[ty * K + tx];
        Bs[ty][tx] = B[ty * N + tx];
        __syncthreads();

        A += BK;
        B += BK * N;

        for (int i = 0; i < BK; ++i) {
            sum += As[ty][i] * Bs[i][tx];
        }
        __syncthreads();
    }
    C[ty * N + tx] = alpha * sum + beta * C[ty * N + tx];
}

int main() {
    std::vector<int> sizes = {128, 256, 512, 1024, 2048, 4096};

    // 打开 csv 文件
    std::ofstream ofs("sgemm_benchmark_v2.csv");
    ofs << "Size,CUBLAS_GFLOPS,MySGEMM_FLOPS,Matched" << std::endl;

    for (int N : sizes) {
        std::cout << "Testing size: " << N << std::endl;
        size_t size = N * N * sizeof(float);
        float* A = (float*)malloc(size);
        float* B = (float*)malloc(size);
        float* C_cublas = (float*)malloc(size);
        float* C_v2 = (float*)malloc(size);

        float* d_A;
        float* d_B;
        float* d_C_v2;
        checkCudaError(cudaMalloc(&d_A, size), "cudaMalloc d_A failed");
        checkCudaError(cudaMalloc(&d_B, size), "cudaMalloc d_B failed");
        checkCudaError(cudaMalloc(&d_C_v2, size), "cudaMalloc d_C_v2 failed");

        bool out_of_memory = false;
        float alpha = 1.0f;
        float beta = 0.0f;
        
        // 初始化输入矩阵 A 和 B
        for (int i = 0; i < N * N; ++i) {
            A[i] = 1.0f;
            B[i] = 2.0f;
        }

        try {
            // 拷贝输入矩阵到 GPU
            checkCudaError(cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice), "cudaMemcpy A to device failed");
            checkCudaError(cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice), "cudaMemcpy A to device failed");

            cublasHandle_t handle;
            checkCublasError(cublasCreate(&handle), "cublasCreate handle failed");

            cudaEvent_t start, end;
            checkCudaError(cudaEventCreate(&start), "cudaEventCreate(start) failed");
            checkCudaError(cudaEventCreate(&end), "cudaEventCreate(end) failed");

            // warm up
            int warmup_times = 10;
            for (int i = 0; i < warmup_times; ++i) {
                checkCublasError(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, d_B, N, d_A, N, &beta, d_C_v2, N), 
                                "warm up cublasSgemm failed");
            }
            cudaDeviceSynchronize();

            // cublas gemm kernel launch
            int repeat_times = 5;
            float cublas_time_ms = 0.0f;
            checkCudaError(cudaEventRecord(start), "cublas cudaEventRecord(start) failed");
            for (int i = 0; i < repeat_times; ++i) {
                checkCublasError(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, d_B, N, d_A, N, &beta, d_C_v2, N), 
                                "cublasSgemm failed");
            }
            checkCudaError(cudaEventRecord(end),  "cublas cudaEventRecord(end) failed");
            checkCudaError(cudaEventSynchronize(end), "cublas cudaEventSynchronize(end) failed");
            checkCudaError(cudaEventElapsedTime(&cublas_time_ms, start, end), "cublas cudaEventElapsedTime failed");

            // 拷贝 cublas 结果
            checkCudaError(cudaMemcpy(C_cublas, d_C_v2, size, cudaMemcpyDeviceToHost), "cudaMemcpy C_cublas failed");

            // mysgemm v2
            checkCudaError(cudaMemset(d_C_v2, 0, size), "cudaMemset d_C_v2 failed");

            // 设定每个 Block 是二维的: 32 x 32 = 1024 个线程
            dim3 thread_num(BLOCK_SIZE, BLOCK_SIZE);

            // 计算需要多少个 Block 才能覆盖整个矩阵
            dim3 block_num((N + thread_num.x - 1) / thread_num.x, (N + thread_num.y - 1) / thread_num.y);

            // warm up
            for (int i = 0; i < warmup_times; ++i) {
                mysgemm_v2<<<block_num, thread_num>>>(N, N, N, alpha, d_A, d_B, beta, d_C_v2);
            }
            cudaDeviceSynchronize();

            // v2 kenrel launch
            float v2_time_ms = 0.0f;
            checkCudaError(cudaEventRecord(start), "v2 cudaEventRecord(start) failed");
            for (int i = 0; i < repeat_times; ++i) {
                mysgemm_v2<<<block_num, thread_num>>>(N, N, N, alpha, d_A, d_B, beta, d_C_v2);
            }
            checkCudaError(cudaEventRecord(end), "v2 cudaEventRecord(end) failed");
            checkCudaError(cudaEventSynchronize(end), "v2 cudaEventSynchronize(end) failed");
            checkCudaError(cudaEventElapsedTime(&v2_time_ms, start, end), "v2 cudaEventElapsedTime failed");

            // 拷贝 v2 结果
            checkCudaError(cudaMemcpy(C_v2, d_C_v2, size, cudaMemcpyDeviceToHost), "cudaMemcpy C_v2 failed");

            // 结果比较
            int error_count = 0;
            for (int i = 0; i < N * N && error_count < 10; ++i) {
                if (fabsf(C_cublas[i] - C_v2[i]) > 1e-5) {
                    ++error_count;
                }
            }

            // GFLOPs: 每秒执行的浮点运算次数
            // 每个 C[gy][gx] 做 K 次乘加运算 (2K 次 FLOPs)
            // 因此总浮点运算次数 (FLOPs) = 2 × M × N × K = 2N^3
            float cublas_gflops = repeat_times * 2.0f * N * N * N / (cublas_time_ms * 1e6f);
            float v2_gflops = repeat_times * 2.0f * N * N * N / (v2_time_ms * 1e6f);

            // 写入 CSV
            ofs << N << "," << cublas_gflops << "," << v2_gflops << "," << (error_count == 0 ? "1" : "0") << std::endl;

            // 释放资源
            cublasDestroy(handle);
            cudaEventDestroy(start);
            cudaEventDestroy(end);
            cudaFree(d_A);
            cudaFree(d_B);
            cudaFree(d_C_v2);
            free(A);
            free(B);
            free(C_cublas);
            free(C_v2);
        } catch (...) {
            std::cerr << "Out of memory or error during testing size: " << N << std::endl;
            out_of_memory = true;
        }
        if (!out_of_memory) {
            std::cout << "Finished size: " << N << std::endl;
        } else {
            ofs << N << ",OOM,OOM,0" << std::endl;
        }
    }

    ofs.close();
    std::cout << "Benchmark completed. Results saved to 'sgemm_benchmark_v2.csv'" << std::endl;

    return 0;
}