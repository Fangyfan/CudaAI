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

constexpr int WARP_SIZE = 32;
#define FLOAT4(p) (*reinterpret_cast<float4*>(&(p)))
#define OFFSET(y, x, ld) ((y) * (ld) + (x))

template<int BLOCK_DIM, int BM, int BN, int BK, int WM, int WN, int WM_ITER, int WN_ITER, int TM, int TN>
__global__ void __launch_bounds__(BLOCK_DIM) mysgemm_v6(int M, int N, int K, float alpha, float* A, float* B, float beta, float* C) {
    int bx = blockIdx.x * BN; // 当前 block 在 gird 内 x 方向的坐标
    int by = blockIdx.y * BM; // 当前 block 在 grid 内 y 方向的坐标

    int warp_id = threadIdx.x / WARP_SIZE; // 当前线程属于第几个 warp
    int wx = warp_id % (BN / WN) * WN; // 当前 warp 在 block 内 x 坐标
    int wy = warp_id / (BN / WN) * WM; // 当前 warp 在 block 内 y 坐标

    constexpr int WM_SUB = WM / WM_ITER; // 该 warp 内的线程实际处理的 sub warp tile 宽度
    constexpr int WN_SUB = WN / WN_ITER; // 该 warp 内的线程实际处理的 sub warp tile 长度

    int lane_id = threadIdx.x % WARP_SIZE; // 当前线程在 warp 内的编号
    int tx = lane_id % (WN_SUB / TN) * TN; // 当前 thread 在 sub warp 内 x 坐标
    int ty = lane_id / (WN_SUB / TN) * TM; // 当前 thread 在 sub warp 内 y 坐标

    // A[M, K] 中 block tile 的左上角 y 坐标为 by，则一维偏移量就是 by * K
    // B[K, N] 中 block tile 的左上角 x 坐标为 bx，则一维偏移量就是 bx
    // C[M, N] 中 warp  tile 的左上角 y 坐标为 by + wy，x 坐标为 bx + wx
    A += by * K;
    B += bx;
    C += (by + wy) * N + (bx + wx);

    __shared__ float As[BK * BM];
    __shared__ float Bs[BK * BN];
    
    int a_tile_x = threadIdx.x % (BK / 4) * 4; // 当前线程负责搬运 A tile 内的 x 坐标
    int a_tile_y = threadIdx.x / (BK / 4);     // 当前线程负责搬运 A tile 内的 y 坐标
    int b_tile_x = threadIdx.x % (BN / 4) * 4; // 当前线程负责搬运 B tile 内的 x 坐标
    int b_tile_y = threadIdx.x / (BN / 4);     // 当前线程负责搬运 B tile 内的 y 坐标

    constexpr int a_tile_stride = (BLOCK_DIM * 4) / BK; // Block 内所有线程每次能搬运 A 多少行
    constexpr int b_tile_stride = (BLOCK_DIM * 4) / BN; // Block 内所有线程每次能搬运 B 多少行

    float temp[TM * WM_ITER * TN * WN_ITER] = { 0.0f }; // 当前线程负责计算 C 中的小 tile 寄存器
    float a_vec[TM * WM_ITER]; // 当前线程读取的 As[ty...ty+TM-1][x] 寄存器
    float b_vec[TN * WN_ITER]; // 当前线程读取的 Bs[x][tx...tx+TN-1] 寄存器

    // 枚举每个 block tile
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
        __syncthreads(); // block 内同步，确保共享内存 As 和 Bs 拷贝完成

        A += BK; 	 // As 向右移动一个 tile，一维偏移量增加 BK
        B += BK * N; // Bs 向下移动一个 tile，一维偏移量增加 BK * N

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
        __syncthreads(); // block 内同步，确保共享内存 As 和 Bs 读取完成，不要和拷贝冲突
    }

    // thread tile 负责 TM × TN 的小 C tile，坐标范围 [ty...ty+TM-1] × [tx...tx+TN-1]
    // 注意 thread tile 迭代了 [WM_ITER, WN_ITER] 次
    for (int sub_y = 0; sub_y < WM_ITER; ++sub_y) {
        for (int sub_x = 0; sub_x < WN_ITER; ++sub_x) {
            float* C_sub = C + OFFSET(sub_y * WM_SUB, sub_x * WN_SUB, N); // 当前 warp sub tile
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

std::vector<int> generate() {
    std::vector<int> sizes;
    for (int i = 256; i <= 8192; i += 256) {
        sizes.push_back(i);
    }
    return sizes;
}

int main() {
    // std::vector<int> sizes = generate();
    std::vector<int> sizes = {128, 256, 512, 1024, 2048, 4096};

    // 打开 csv 文件
    std::ofstream ofs("sgemm_benchmark_v6.csv");
    ofs << "Size,CUBLAS_GFLOPS,MySGEMM_FLOPS,UnMatched,Ratio" << std::endl;

    for (int N : sizes) {
        std::cout << "Testing size: " << N << std::endl;
        size_t size = N * N * sizeof(float);
        float* A = (float*)malloc(size);
        float* B = (float*)malloc(size);
        float* C_cublas = (float*)malloc(size);
        float* C_v6 = (float*)malloc(size);

        float* d_A;
        float* d_B;
        float* d_C_v6;
        checkCudaError(cudaMalloc(&d_A, size), "cudaMalloc d_A failed");
        checkCudaError(cudaMalloc(&d_B, size), "cudaMalloc d_B failed");
        checkCudaError(cudaMalloc(&d_C_v6, size), "cudaMalloc d_C_v6 failed");

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
                checkCublasError(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, d_B, N, d_A, N, &beta, d_C_v6, N), 
                                "warm up cublasSgemm failed");
            }
            cudaDeviceSynchronize();

            // cublas gemm kernel launch
            int repeat_times = 5;
            float cublas_time_ms = 0.0f;
            checkCudaError(cudaEventRecord(start), "cublas cudaEventRecord(start) failed");
            for (int i = 0; i < repeat_times; ++i) {
                checkCublasError(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, d_B, N, d_A, N, &beta, d_C_v6, N), 
                                "cublasSgemm failed");
            }
            checkCudaError(cudaEventRecord(end),  "cublas cudaEventRecord(end) failed");
            checkCudaError(cudaEventSynchronize(end), "cublas cudaEventSynchronize(end) failed");
            checkCudaError(cudaEventElapsedTime(&cublas_time_ms, start, end), "cublas cudaEventElapsedTime failed");

            // 拷贝 cublas 结果
            checkCudaError(cudaMemcpy(C_cublas, d_C_v6, size, cudaMemcpyDeviceToHost), "cudaMemcpy C_cublas failed");

            // mysgemm v6
            checkCudaError(cudaMemset(d_C_v6, 0, size), "cudaMemset d_C_v6 failed");

            // 每个 Block 负责计算 C 矩阵的一个大 tile: 128 × 128
            constexpr int BLOCK_DIM = 128;
            constexpr int BN = 128;
            constexpr int BM = 128;
            constexpr int BK = 16;
            // 每个 Warp 负责计算 C 矩阵的一个中 tile: 64 × 64
            constexpr int WM = 64;
            constexpr int WN = 64;
            constexpr int WN_ITER = 4;
            // 每个 thread 计算 C 矩阵中的一个小 tile: 8 × 4
            constexpr int TM = 8;
            constexpr int TN = 4;
            static_assert((WM * WN) % (WARP_SIZE * TM * TN * WN_ITER) == 0);
            constexpr int WM_ITER = (WM * WN) / (WARP_SIZE * TM * TN * WN_ITER); // 处理 1 个 warp 的迭代次数

            static_assert(BN % WN == 0);
            static_assert(WM % WM_ITER == 0);
            static_assert(WN % WN_ITER == 0);
            static_assert((WN / WN_ITER) % TN == 0);
            static_assert(BN % 4 == 0);
            static_assert(BK % 4 == 0);
            static_assert((4 * BLOCK_DIM) % BK == 0);
            static_assert((4 * BLOCK_DIM) % BN == 0);

            // 每个 Block 内包含 128 个线程
            dim3 blockDim(BLOCK_DIM);
            // 计算至少要多少个 Block 才能覆盖 N × N 完整的 C 矩阵
            dim3 gridDim(N / BN, N / BM);

            // warm up
            for (int i = 0; i < warmup_times; ++i) {
                mysgemm_v6<BLOCK_DIM, BM, BN, BK, WM, WN, WM_ITER, WN_ITER, TM, TN><<<gridDim, blockDim>>>(
                    N, N, N, alpha, d_A, d_B, beta, d_C_v6
                );
            }
            cudaDeviceSynchronize();

            // v6 kenrel launch
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

            // 拷贝 v6 结果
            checkCudaError(cudaMemcpy(C_v6, d_C_v6, size, cudaMemcpyDeviceToHost), "cudaMemcpy C_v6 failed");

            // 结果比较
            int error_count = 0;
            float max_rel_err = 0.0f; // 用于记录最大相对误差
            for (int i = 0; i < N * N; ++i) {
                float ref = C_cublas[i];
                float val = C_v6[i];
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
            float v6_gflops = repeat_times * 2.0f * N * N * N / (v6_time_ms * 1e6f);

            // 写入 CSV
            ofs << N << "," << cublas_gflops << "," << v6_gflops << "," << error_count << ",";
            ofs << std::fixed << std::setprecision(2) << (100 * v6_gflops / cublas_gflops) << "%" << std::endl;

            // 释放资源
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
        if (!out_of_memory) {
            std::cout << "Finished size: " << N << std::endl;
        } else {
            ofs << N << ",OOM,OOM,0" << std::endl;
        }
    }

    ofs.close();
    std::cout << "Benchmark completed. Results saved to 'sgemm_benchmark_v6.csv'" << std::endl;

    return 0;
}