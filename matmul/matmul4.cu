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

#define FLOAT4(p) (*reinterpret_cast<float4*>(&(p)))
#define OFFSET(y, x, ld) ((y) * (ld) + (x))

// C[M, N] = A[M, K] × B[K, N]
template<int BLOCK_DIM, int BM, int BN, int BK, int TM, int TN>
__global__ void __launch_bounds__(BLOCK_DIM) mysgemm_v5(int M, int N, int K, float alpha, float* A, float* B, float beta, float* C) {
    int bx = blockIdx.x * BN; // 当前 Block 在 x 方向的坐标
    int by = blockIdx.y * BM; // 当前 Block 在 y 方向的坐标
    int tx = threadIdx.x * TN; // Thread tile 左上角 x 坐标 [0, 16] * 8
    int ty = threadIdx.y * TM; // Thread tile 左上角 y 坐标 [0, 16] * 8
    
    // A[M, K] 中 tile 的左上角 y 坐标为 by，则一维偏移量就是 by * K
    // B[K, N] 中 tile 的左上角 x 坐标为 bx，则一维偏移量就是 bx
    // C[M, N] 中 tile 的左上角 y 坐标为 by，x 坐标为 bx，则一维偏移量就是 by * N + bx
    A += by * K;
    B += bx;
    C += by * N + bx;
    
    // 双缓冲流水线
    __shared__ float As[2][BK * BM]; // A Tile Size = (8, 128)
    __shared__ float Bs[2][BK * BN]; // B Tile Size = (8, 128)
    
    int tid = threadIdx.y * blockDim.x + threadIdx.x; // 当前线程在 Block 内的一维偏移量
    int a_tile_x = tid % (BK / 4) * 4; // 当前线程负责搬运 A Tile 内的 x 坐标
    int a_tile_y = tid / (BK / 4);     // 当前线程负责搬运 A Tile 内的 y 坐标
    int b_tile_x = tid % (BN / 4) * 4; // 当前线程负责搬运 B Tile 内的 x 坐标
    int b_tile_y = tid / (BN / 4);     // 当前线程负责搬运 B Tile 内的 y 坐标

    constexpr int a_tile_stride = (BLOCK_DIM * 4) / BK; // Block 内所有线程每次能搬运 A 多少行
    constexpr int b_tile_stride = (BLOCK_DIM * 4) / BN; // Block 内所有线程每次能搬运 B 多少行

    float temp[TM][TN] = { 0.0f }; // 当前线程负责计算 C 中的小 Tile 寄存器

    // 寄存器级流水线
    float a_reg[4 * BM / a_tile_stride]; // 用于临时存储 A Tile_k+1 寄存器
    float b_reg[4 * BK / b_tile_stride]; // 用于临时存储 B Tile_k+1 寄存器
    float a_vec[2][TM]; // 当前线程读取的 As[0/1][ty...ty+TM-1][x] 寄存器
    float b_vec[2][TN]; // 当前线程读取的 Bs[0/1][x][tx...tx+TN-1] 寄存器

    // 第一次加载: 从全局内存读取第 0 个 Tile 的数据，存入 As[0] 和 Bs[0]
#pragma unroll
    for (int i = 0, j = 0; i < BM; i += a_tile_stride, j += 4) {
        FLOAT4(a_reg[j]) = FLOAT4(A[OFFSET(a_tile_y + i, a_tile_x, K)]);
        As[0][OFFSET(a_tile_x, a_tile_y + i, BM)] = a_reg[j];
        As[0][OFFSET(a_tile_x + 1, a_tile_y + i, BM)] = a_reg[j + 1];
        As[0][OFFSET(a_tile_x + 2, a_tile_y + i, BM)] = a_reg[j + 2];
        As[0][OFFSET(a_tile_x + 3, a_tile_y + i, BM)] = a_reg[j + 3];
    }
#pragma unroll
    for (int i = 0; i < BK; i += b_tile_stride) {
        FLOAT4(Bs[0][OFFSET(b_tile_y + i, b_tile_x, BN)])
        = FLOAT4(B[OFFSET(b_tile_y + i, b_tile_x, N)]);
    }
    __syncthreads(); // 必须同步！确保所有线程都把数据搬运到 Shared 后，任何线程才能开始读取 Shared

    // 预加载寄存器: 从 Shared Memory (As[0], Bs[0]) 中拿出第 0 个 Tile，放到寄存器中准备计算
    // 只存 x=0 所需的 TM × 1 和 1 × TN，即 As[0][ty...ty+TM-1][x] 和 Bs[0][x][tx...tx+TN-1]
#pragma unroll
    for (int i = 0; i < TM; i += 4) {
        FLOAT4(a_vec[0][i]) = FLOAT4(As[0][OFFSET(0, ty + i, BM)]);
    }
#pragma unroll
    for (int j = 0; j < TN; j += 4) {
        FLOAT4(b_vec[0][j]) = FLOAT4(Bs[0][OFFSET(0, tx + j, BN)]);
    }

    int read_type;
    int write_type = 1;

    // 枚举每个 Tile_k 进行计算
    for (int k = 0; k < K; k += BK) {
        // 此时我们要计算 Tile_k，但立刻发起了读取 Tile_k+1 的指令
        // 注意：数据被读到了寄存器 a/b reg 里，而不是 Shared Memory！
        // 为什么要暂存寄存器？因为 Shared Memory 现在正忙着给计算单元喂数据（读写冲突），不能被打扰
        if (k + BK < K) {
            A += BK; 	 // As 向右移动一个 Tile，一维偏移量增加 BK
            B += BK * N; // Bs 向下移动一个 Tile，一维偏移量增加 BK * N
#pragma unroll
            for (int i = 0, j = 0; i < BM; i += a_tile_stride, j += 4) {
                FLOAT4(a_reg[j]) = FLOAT4(A[OFFSET(a_tile_y + i, a_tile_x, K)]);
            }
#pragma unroll
            for (int i = 0, j = 0; i < BK; i += b_tile_stride, j += 4) {
                FLOAT4(b_reg[j]) = FLOAT4(B[OFFSET(b_tile_y + i, b_tile_x, N)]);
            }
        }
        read_type = write_type ^ 1;

        // 遍历 Tile_k 计算部分点积 temp[i][j]
        // 注意 x < BK - 1，因为要预加载 As[0][ty...ty+TM-1][x+1] 和 Bs[0][x+1][tx...tx+TN-1] 到 a/b vec
#pragma unroll
        for (int x = 0; x < BK - 1; ++x) {
            // 此时正在准备计算第 x 行，但先把第 x+1 行的数据从 Shared Memory 读到寄存器 vec
            // 这里的 (x + 1) % 2 是为了在两个 vec 寄存器组之间轮转
#pragma unroll
            for (int i = 0; i < TM; i += 4) { // 把 As[ty ... ty+TM-1][x+1] 这一列的数取出来
                FLOAT4(a_vec[(x + 1) & 1][i]) = FLOAT4(As[read_type][OFFSET(x + 1, ty + i, BM)]);
            }
#pragma unroll
            for (int j = 0; j < TN; j += 4) { // 把 Bs[x+1][tx ... tx+TN-1] 这一行的数取出来
                FLOAT4(b_vec[(x + 1) & 1][j]) = FLOAT4(Bs[read_type][OFFSET(x + 1, tx + j, BN)]);
            }
            // 注意：它用的是 vec[x % 2]，这是 "上一次循环" 就已经预取好的数据！
            // 这样，计算指令 (FFMA) 就不需要等待 Shared Memory 的读取延迟，因为数据早就已经在寄存器里了
#pragma unroll
            for (int i = 0; i < TM; ++i) {
#pragma unroll
                for (int j = 0; j < TN; ++j) {
                    temp[i][j] += a_vec[x & 1][i] * b_vec[x & 1][j]; // 部分点积: As[ty + i][x] * Bs[x][tx + j]
                }
            }
        }

        if (k + BK < K) {
            // 把暂存在 a/b reg 里的数据，写入另一个空闲的 Shared Memory (write_type)
#pragma unroll
            for (int i = 0, j = 0; i < BM; i += a_tile_stride, j += 4) {
                As[write_type][OFFSET(a_tile_x, a_tile_y + i, BM)] = a_reg[j];
                As[write_type][OFFSET(a_tile_x + 1, a_tile_y + i, BM)] = a_reg[j + 1];
                As[write_type][OFFSET(a_tile_x + 2, a_tile_y + i, BM)] = a_reg[j + 2];
                As[write_type][OFFSET(a_tile_x + 3, a_tile_y + i, BM)] = a_reg[j + 3];
            }
#pragma unroll
            for (int i = 0, j = 0; i < BK; i += b_tile_stride, j += 4) {
                FLOAT4(Bs[write_type][OFFSET(b_tile_y + i, b_tile_x, BN)]) = FLOAT4(b_reg[j]);
            }

            // 必须同步！确保写入共享内存 As 和 Bs 完成，不要和下面的读取冲突
            __syncthreads();

            // 为了让下一轮循环 (k += BK) 的 x=0 能跑起来，这里必须手动预加载下一轮的第 0 行数据！
#pragma unroll
            for (int i = 0; i < TM; i += 4) {
                FLOAT4(a_vec[0][i]) = FLOAT4(As[write_type][OFFSET(0, ty + i, BM)]);
            }
#pragma unroll
            for (int j = 0; j < TN; j += 4) {
                FLOAT4(b_vec[0][j]) = FLOAT4(Bs[write_type][OFFSET(0, tx + j, BN)]);
            }

            write_type ^= 1; // 切换乒乓缓冲下标：读的变写的，写的变读的
        }

        // 不限条件，补上 x = BK - 1 这一行的计算，此时用的数据是之前 x = BK - 2 时预加载的 x+1 数据
#pragma unroll
        for (int i = 0; i < TM; ++i) {
#pragma unroll
            for (int j = 0; j < TN; ++j) {
                temp[i][j] += a_vec[(BK - 1) & 1][i] * b_vec[(BK - 1) & 1][j];
            }
        }
    }

    // Thread tile 负责 TM × TN 的小 C tile，坐标范围 [ty...ty+TM-1] × [tx...tx+TN-1]
#pragma unroll
    for (int i = 0; i < TM; ++i) {
#pragma unroll
        for (int j = 0; j < TN; j += 4) {
            float4 c4 = FLOAT4(C[OFFSET(ty + i, tx + j, N)]);
            FLOAT4(C[OFFSET(ty + i, tx + j, N)]) = make_float4(
                alpha * temp[i][j] + beta * c4.x, 
                alpha * temp[i][j + 1] + beta * c4.y, 
                alpha * temp[i][j + 2] + beta * c4.z, 
                alpha * temp[i][j + 3] + beta * c4.w
            );
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
    std::ofstream ofs("sgemm_benchmark_v5.csv");
    ofs << "Size,CUBLAS_GFLOPS,MySGEMM_FLOPS,UnMatched,Ratio" << std::endl;

    for (int N : sizes) {
        std::cout << "Testing size: " << N << std::endl;
        size_t size = N * N * sizeof(float);
        float* A = (float*)malloc(size);
        float* B = (float*)malloc(size);
        float* C_cublas = (float*)malloc(size);
        float* C_v5 = (float*)malloc(size);

        float* d_A;
        float* d_B;
        float* d_C_v5;
        checkCudaError(cudaMalloc(&d_A, size), "cudaMalloc d_A failed");
        checkCudaError(cudaMalloc(&d_B, size), "cudaMalloc d_B failed");
        checkCudaError(cudaMalloc(&d_C_v5, size), "cudaMalloc d_C_v5 failed");

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
                checkCublasError(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, d_B, N, d_A, N, &beta, d_C_v5, N), 
                                "warm up cublasSgemm failed");
            }
            cudaDeviceSynchronize();

            // cublas gemm kernel launch
            int repeat_times = 5;
            float cublas_time_ms = 0.0f;
            checkCudaError(cudaEventRecord(start), "cublas cudaEventRecord(start) failed");
            for (int i = 0; i < repeat_times; ++i) {
                checkCublasError(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, d_B, N, d_A, N, &beta, d_C_v5, N), 
                                "cublasSgemm failed");
            }
            checkCudaError(cudaEventRecord(end),  "cublas cudaEventRecord(end) failed");
            checkCudaError(cudaEventSynchronize(end), "cublas cudaEventSynchronize(end) failed");
            checkCudaError(cudaEventElapsedTime(&cublas_time_ms, start, end), "cublas cudaEventElapsedTime failed");

            // 拷贝 cublas 结果
            checkCudaError(cudaMemcpy(C_cublas, d_C_v5, size, cudaMemcpyDeviceToHost), "cudaMemcpy C_cublas failed");

            // mysgemm v5
            checkCudaError(cudaMemset(d_C_v5, 0, size), "cudaMemset d_C_v5 failed");

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
                mysgemm_v5<256, 128, 128, 8, 8, 8><<<gridDim, blockDim>>>(N, N, N, alpha, d_A, d_B, beta, d_C_v5);
            }
            cudaDeviceSynchronize();

            // v5 kenrel launch
            float v5_time_ms = 0.0f;
            checkCudaError(cudaEventRecord(start), "v5 cudaEventRecord(start) failed");
            for (int i = 0; i < repeat_times; ++i) {
                mysgemm_v5<256, 128, 128, 8, 8, 8><<<gridDim, blockDim>>>(N, N, N, alpha, d_A, d_B, beta, d_C_v5);
            }
            checkCudaError(cudaEventRecord(end), "v5 cudaEventRecord(end) failed");
            checkCudaError(cudaEventSynchronize(end), "v5 cudaEventSynchronize(end) failed");
            checkCudaError(cudaEventElapsedTime(&v5_time_ms, start, end), "v5 cudaEventElapsedTime failed");

            // 拷贝 v5 结果
            checkCudaError(cudaMemcpy(C_v5, d_C_v5, size, cudaMemcpyDeviceToHost), "cudaMemcpy C_v5 failed");

            // 结果比较
            int error_count = 0;
            float max_rel_err = 0.0f; // 用于记录最大相对误差
            for (int i = 0; i < N * N; ++i) {
                float ref = C_cublas[i];
                float val = C_v5[i];
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
            float v5_gflops = repeat_times * 2.0f * N * N * N / (v5_time_ms * 1e6f);

            // 写入 CSV
            ofs << N << "," << cublas_gflops << "," << v5_gflops << "," << error_count << ",";
            ofs << std::fixed << std::setprecision(2) << (100 * v5_gflops / cublas_gflops) << "%" << std::endl;

            // 释放资源
            cublasDestroy(handle);
            cudaEventDestroy(start);
            cudaEventDestroy(end);
            cudaFree(d_A);
            cudaFree(d_B);
            cudaFree(d_C_v5);
            free(A);
            free(B);
            free(C_cublas);
            free(C_v5);
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
    std::cout << "Benchmark completed. Results saved to 'sgemm_benchmark_v5.csv'" << std::endl;

    return 0;
}