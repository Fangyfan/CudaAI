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

#define FLOAT4(p) (reinterpret_cast<float4*>(&(p)))
#define OFFSET(y, x, ld) ((y) * (ld) + (x))

// C[M, N] = A[M, K] × B[K, N]
template<int BLOCK_DIM, int BM, int BN, int BK, int TM, int TN>
__global__ void mysgemm_v4(int M, int N, int K, float alpha, float* A, float* B, float beta, float* C) {
    int bx = blockIdx.x; // 当前 Block 在 x 方向的编号
    int by = blockIdx.y; // 当前 Block 在 y 方向的编号
    int tx = threadIdx.x * TN; // Thread tile 左上角 x 坐标 [0, 16] * 8
    int ty = threadIdx.y * TM; // Thread tile 左上角 y 坐标 [0, 16] * 8
    
    A = &A[by * BM * K]; // A[M, K] 中 tile 的左上角 y 坐标为 by * BM，则一维偏移量就是 by * BM * K
    B = &B[bx * BN];     // B[K, N] 中 tile 的左上角 x 坐标为 bx * BN，则一维偏移量就是 bx * BN

    // C[M, N] 中 tile 的左上角 y 坐标为 by * BM，x 坐标为 bx * BN
    // 则一维偏移量就是 by * BM * N + bx * BN
    C = &C[by * BM * N + bx * BN];
    
    __shared__ float As[BM][BK]; // A tile size = (128, 8)
    __shared__ float Bs[BK][BN]; // B tile size = (8, 128)
    
    int tid = threadIdx.y * blockDim.x + threadIdx.x; // 当前线程在 Block 内的一维偏移量
    int a_tile_x = tid % BK; // 当前线程负责搬运 A tile 内的 x 坐标
    int a_tile_y = tid / BK; // 当前线程负责搬运 A tile 内的 y 坐标
    constexpr int a_tile_stride = BLOCK_DIM / BK; // Block 内所有线程每次能搬运多少行
    int b_tile_x = tid % BN; // 当前线程负责搬运 B tile 内的 x 坐标
    int b_tile_y = tid / BN; // 当前线程负责搬运 B tile 内的 y 坐标
    constexpr int b_tile_stride = BLOCK_DIM / BN; // Block 内所有线程每次能搬运多少行

    float temp[TM][TN] = { 0.0f }; // 当前线程负责计算 C 中的小 tile 寄存器

    // 枚举每个 A/B tile
    for (int k = 0; k < K; k += BK) {
#pragma unroll
        for (int i = 0; i < BM; i += a_tile_stride) {
            As[i + a_tile_y][a_tile_x] = A[(i + a_tile_y) * K + a_tile_x]; // 搬运 shared As
        }
#pragma unroll
        for (int i = 0; i < BK; i += b_tile_stride) {
            Bs[i + b_tile_y][b_tile_x] = B[(i + b_tile_y) * N + b_tile_x]; // 搬运 shared Bs
        }
        __syncthreads(); // block 内同步，确保共享内存 As 和 Bs 搬运完成
        
        A += BK; 	 // As 向右移动一个 tile，一维偏移量增加 BK
        B += BK * N; // Bs 向下移动一个 tile，一维偏移量增加 BK * N

        // 遍历 Thread tile 内部 C 元素 temp[i][j]，计算部分点积
#pragma unroll
    for (int x = 0; x < BK; ++x) {
#pragma unroll
        for (int i = 0; i < TM; ++i) {
#pragma unroll
            for (int j = 0; j < TN; ++j) {
                    temp[i][j] += As[ty + i][x] * Bs[x][tx + j];
                }
            }
        }
        __syncthreads(); // block 内同步，读取共享内存 As 和 Bs 完成，不要和搬运冲突
    }

    // Thread tile 负责 TM × TN 的小 C tile，坐标范围 [ty...ty+TM-1] × [tx...tx+TN-1]
#pragma unroll
    for (int i = 0; i < TM; ++i) {
#pragma unroll
        for (int j = 0; j < TN; ++j) {
            C[(ty + i) * N + (tx + j)] = alpha * temp[i][j] + beta * C[(ty + i) * N + (tx + j)];
        }
    }
}

int main() {
    std::vector<int> sizes = {128, 256, 512, 1024, 2048, 4096};

    // 打开 csv 文件
    std::ofstream ofs("sgemm_benchmark_v4.csv");
    ofs << "Size,CUBLAS_GFLOPS,MySGEMM_FLOPS,UnMatched,Ratio" << std::endl;

    for (int N : sizes) {
        std::cout << "Testing size: " << N << std::endl;
        size_t size = N * N * sizeof(float);
        float* A = (float*)malloc(size);
        float* B = (float*)malloc(size);
        float* C_cublas = (float*)malloc(size);
        float* C_v4 = (float*)malloc(size);

        float* d_A;
        float* d_B;
        float* d_C_v4;
        checkCudaError(cudaMalloc(&d_A, size), "cudaMalloc d_A failed");
        checkCudaError(cudaMalloc(&d_B, size), "cudaMalloc d_B failed");
        checkCudaError(cudaMalloc(&d_C_v4, size), "cudaMalloc d_C_v4 failed");

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
                checkCublasError(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, d_B, N, d_A, N, &beta, d_C_v4, N), 
                                "warm up cublasSgemm failed");
            }
            cudaDeviceSynchronize();

            // cublas gemm kernel launch
            int repeat_times = 5;
            float cublas_time_ms = 0.0f;
            checkCudaError(cudaEventRecord(start), "cublas cudaEventRecord(start) failed");
            for (int i = 0; i < repeat_times; ++i) {
                checkCublasError(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, d_B, N, d_A, N, &beta, d_C_v4, N), 
                                "cublasSgemm failed");
            }
            checkCudaError(cudaEventRecord(end),  "cublas cudaEventRecord(end) failed");
            checkCudaError(cudaEventSynchronize(end), "cublas cudaEventSynchronize(end) failed");
            checkCudaError(cudaEventElapsedTime(&cublas_time_ms, start, end), "cublas cudaEventElapsedTime failed");

            // 拷贝 cublas 结果
            checkCudaError(cudaMemcpy(C_cublas, d_C_v4, size, cudaMemcpyDeviceToHost), "cudaMemcpy C_cublas failed");

            // mysgemm v4
            checkCudaError(cudaMemset(d_C_v4, 0, size), "cudaMemset d_C_v4 failed");

            // 每个 Block 负责计算 C 矩阵的一个大 tile: 128 × 128
            constexpr int BN = 128;
            constexpr int BM = 128;
            // 每个 thread 计算 C 矩阵中的一个小 tile: 8 × 8
            constexpr int TM = 8;
            constexpr int TN = 8;
            // 每个 Block 是二维的: (128 / 8) x (128 / 8) = 16 × 16 = 256 个线程
            dim3 blockDim(BN / TN, BM / TM);
            // 计算至少要多少个 Block 才能覆盖 N × N 完整的 C 矩阵
            dim3 gridDim((N + BN - 1) / BN, (N + BM - 1) / BM);

            // warm up
            for (int i = 0; i < warmup_times; ++i) {
                mysgemm_v4<256, 128, 128, 8, 8, 8><<<gridDim, blockDim>>>(N, N, N, alpha, d_A, d_B, beta, d_C_v4);
            }
            cudaDeviceSynchronize();

            // v4 kenrel launch
            float v4_time_ms = 0.0f;
            checkCudaError(cudaEventRecord(start), "v4 cudaEventRecord(start) failed");
            for (int i = 0; i < repeat_times; ++i) {
                mysgemm_v4<256, 128, 128, 8, 8, 8><<<gridDim, blockDim>>>(N, N, N, alpha, d_A, d_B, beta, d_C_v4);
            }
            checkCudaError(cudaEventRecord(end), "v4 cudaEventRecord(end) failed");
            checkCudaError(cudaEventSynchronize(end), "v4 cudaEventSynchronize(end) failed");
            checkCudaError(cudaEventElapsedTime(&v4_time_ms, start, end), "v4 cudaEventElapsedTime failed");

            // 拷贝 v4 结果
            checkCudaError(cudaMemcpy(C_v4, d_C_v4, size, cudaMemcpyDeviceToHost), "cudaMemcpy C_v4 failed");

            // 结果比较
            int error_count = 0;
            float max_rel_err = 0.0f; // 用于记录最大相对误差
            for (int i = 0; i < N * N; ++i) {
                float ref = C_cublas[i];
                float val = C_v4[i];
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
            float v4_gflops = repeat_times * 2.0f * N * N * N / (v4_time_ms * 1e6f);

            // 写入 CSV
            ofs << N << "," << cublas_gflops << "," << v4_gflops << "," << error_count << ",";
            ofs << std::fixed << std::setprecision(2) << (100 * v4_gflops / cublas_gflops) << "%" << std::endl;

            // 释放资源
            cublasDestroy(handle);
            cudaEventDestroy(start);
            cudaEventDestroy(end);
            cudaFree(d_A);
            cudaFree(d_B);
            cudaFree(d_C_v4);
            free(A);
            free(B);
            free(C_cublas);
            free(C_v4);
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
    std::cout << "Benchmark completed. Results saved to 'sgemm_benchmark_v4.csv'" << std::endl;

    return 0;
}