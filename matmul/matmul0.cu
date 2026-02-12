#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <iomanip>  // for setprecision
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

// C[M, N] = A[M, K] × B[K, N]
__global__ void mysgemm_v1(int M, int N, int K, float alpha, float* A, float* B, float beta, float* C) {
    // --- 1. 定位身份 ---
    // 计算当前线程在整个网格中的二维坐标
    // blockIdx: 当前在第几个块 (包括 x, y 两个方向)
    // blockDim: 每个块有多大，即每个块有多少个线程 (包括 x, y 两个方向)
    // threadIdx: 当前线程在块里的编号 (包括 x, y 两个方向)
    int gy = blockIdx.y * blockDim.y + threadIdx.y; // 负责结果矩阵的第几行 (对应 M)
    int gx = blockIdx.x * blockDim.x + threadIdx.x; // 负责结果矩阵的第几列 (对应 N)

    // --- 2. 边界检查 (非常重要) ---
    // 因为我们启动线程通常是 32 的倍数（比如 32 x 32 的块）
    // 但矩阵可能是 30 x 30 的
    // 多出来的线程如果不拦住，会去读写非法内存，导致报错
    if (gy >= M || gx >= N) return;

    // --- 3. 核心计算 (点积) ---
    float sum = 0.0f;
    for (int i = 0; i < K; ++i) {
        // A[gy][i] = A[gy * K + i]:
        // A 的宽度是 K (列数)
        // 要找第 gy 行，先跳过前 gy 行（即 gy * K 个元素）
        // 再加上当前循环的偏移量 i

        // B[i][gx] = B[i * N + gx]:
        // B 的宽度是 N (列数)
        // 要找第 i 行（因为我们在遍历 B 的列，实际上是沿着 B 的行向下走）
        // 先跳过前 i 行（即 i * N 个元素）
        // 再加上固定的列号 gx
        sum += A[gy * K + i] * B[i * N + gx];
    }

    // --- 4. 写回结果 ---
    // 计算最终值：C = alpha * (AB) + beta * C
    // 同样使用一维地址映射 C[gy][gx] = C[gy * N + gx]
    C[gy * N + gx] = alpha * sum + beta * C[gy * N + gx];
}

int main() {
    std::vector<int> sizes = {128, 256, 512, 1024, 2048, 4096};

    // 打开 csv 文件
    std::ofstream ofs("sgemm_benchmark_v1.csv");
    ofs << "Size,CUBLAS_GFLOPS,MySGEMM_FLOPS,UnMatched,Ratio" << std::endl;

    for (int N : sizes) {
        std::cout << "Testing size: " << N << std::endl;
        size_t size = N * N * sizeof(float);
        float* A = (float*)malloc(size);
        float* B = (float*)malloc(size);
        float* C_cublas = (float*)malloc(size);
        float* C_v1 = (float*)malloc(size);

        float* d_A;
        float* d_B;
        float* d_C_v1;
        checkCudaError(cudaMalloc(&d_A, size), "cudaMalloc d_A failed");
        checkCudaError(cudaMalloc(&d_B, size), "cudaMalloc d_B failed");
        checkCudaError(cudaMalloc(&d_C_v1, size), "cudaMalloc d_C_v1 failed");

        bool out_of_memory = false;
        float alpha = 1.0f;
        float beta = 0.0f;
        
        // 初始化输入矩阵 A 和 B
        for (int i = 0; i < N * N; ++i) {
            // 生成 0.0 到 1.0 之间的随机浮点数
            A[i] = (float)rand() / RAND_MAX;
            B[i] = (float)rand() / RAND_MAX;
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
                checkCublasError(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, d_B, N, d_A, N, &beta, d_C_v1, N), 
                                "warm up cublasSgemm failed");
            }
            cudaDeviceSynchronize();

            // cublas gemm kernel launch
            int repeat_times = 5;
            float cublas_time_ms = 0.0f;
            checkCudaError(cudaEventRecord(start), "cublas cudaEventRecord(start) failed");
            for (int i = 0; i < repeat_times; ++i) {
                checkCublasError(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, d_B, N, d_A, N, &beta, d_C_v1, N), 
                                "cublasSgemm failed");
            }
            checkCudaError(cudaEventRecord(end),  "cublas cudaEventRecord(end) failed");
            checkCudaError(cudaEventSynchronize(end), "cublas cudaEventSynchronize(end) failed");
            checkCudaError(cudaEventElapsedTime(&cublas_time_ms, start, end), "cublas cudaEventElapsedTime failed");

            // 拷贝 cublas 结果
            checkCudaError(cudaMemcpy(C_cublas, d_C_v1, size, cudaMemcpyDeviceToHost), "cudaMemcpy C_cublas failed");

            // mysgemm v1
            checkCudaError(cudaMemset(d_C_v1, 0, size), "cudaMemset d_C_v1 failed");

            // 每个 Block 负责计算 C 矩阵的一个 tile: 32 × 32
            constexpr int BN = 32;
            constexpr int BM = 32;
            // 每个 thread 计算 C 矩阵中的一个值
            // 每个 Block 是二维的: (32 / 1) x (32 / 1) = 32 × 32 = 1024 个线程
            dim3 blockDim(BN, BM);
            // 计算至少要多少个 Block 才能覆盖 N × N 完整的 C 矩阵
            dim3 gridDim((N + BN - 1) / BN, (N + BM - 1) / BM);

            // warm up
            for (int i = 0; i < warmup_times; ++i) {
                mysgemm_v1<<<gridDim, blockDim>>>(N, N, N, alpha, d_A, d_B, beta, d_C_v1);
            }
            cudaDeviceSynchronize();

            // v1 kenrel launch
            float v1_time_ms = 0.0f;
            checkCudaError(cudaEventRecord(start), "v1 cudaEventRecord(start) failed");
            for (int i = 0; i < repeat_times; ++i) {
                mysgemm_v1<<<gridDim, blockDim>>>(N, N, N, alpha, d_A, d_B, beta, d_C_v1);
            }
            checkCudaError(cudaEventRecord(end), "v1 cudaEventRecord(end) failed");
            checkCudaError(cudaEventSynchronize(end), "v1 cudaEventSynchronize(end) failed");
            checkCudaError(cudaEventElapsedTime(&v1_time_ms, start, end), "v1 cudaEventElapsedTime failed");

            // 拷贝 v1 结果
            checkCudaError(cudaMemcpy(C_v1, d_C_v1, size, cudaMemcpyDeviceToHost), "cudaMemcpy C_v1 failed");

            // 结果比较
            int error_count = 0;
            float max_rel_err = 0.0f; // 用于记录最大相对误差
            for (int i = 0; i < N * N; ++i) {
                float ref = C_cublas[i];
                float val = C_v1[i];
                float diff = fabsf(ref - val);
                
                // 相对误差公式： |diff| / |ref|
                // 添加一个小量 1e-7 防止除以 0
                float rel_err = diff / (fabsf(ref) + 1e-7f);
                
                if (rel_err > max_rel_err) {
                    max_rel_err = rel_err;
                }

                // 允许 1e-4 (0.01%) 的相对误差
                // 对于单精度浮点数矩阵乘法，累加误差是正常的
                if (rel_err > 1e-4) {
                    error_count++;
                }
                
                if (error_count >= 10) break; // 发现10个错误就停止
            }

            // GFLOPs: 每秒执行的浮点运算次数
            // 每个 C[gy][gx] 做 K 次乘加运算 (2K 次 FLOPs)
            // 因此总浮点运算次数 (FLOPs) = 2 × M × N × K = 2N^3
            float cublas_gflops = repeat_times * 2.0f * N * N * N / (cublas_time_ms * 1e6f);
            float v1_gflops = repeat_times * 2.0f * N * N * N / (v1_time_ms * 1e6f);

            // 写入 CSV
            ofs << N << "," << cublas_gflops << "," << v1_gflops << "," << error_count << ",";
            ofs << std::fixed << std::setprecision(2) << (100 * v1_gflops / cublas_gflops) << "%" << std::endl;

            // 释放资源
            cublasDestroy(handle);
            cudaEventDestroy(start);
            cudaEventDestroy(end);
            cudaFree(d_A);
            cudaFree(d_B);
            cudaFree(d_C_v1);
            free(A);
            free(B);
            free(C_cublas);
            free(C_v1);
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
    std::cout << "Benchmark completed. Results saved to 'sgemm_benchmark_v1.csv'" << std::endl;

    return 0;
}