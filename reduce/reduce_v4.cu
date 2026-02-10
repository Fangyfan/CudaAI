#include <cuda_runtime.h>
#include <chrono>
#include <iostream>
#include <vector>

constexpr int N = 1024 * 1024;
constexpr int thread_num = 1024;
constexpr int block_num = N / thread_num;

__device__ __forceinline__ float block_reduce_sum(float val) {
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int warp_num = blockDim.x >> 5;

#pragma unroll
    for (int offset = (warpSize >> 1); offset; offset >>= 1) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }

    __shared__ float shared[32];
    if (lane == 0) {
        shared[warp] = val;
    }
    __syncthreads();

    if (warp == 0) {
        val = threadIdx.x < warp_num ? shared[threadIdx.x] : 0.0f;
#pragma unroll
        for (int offset = (warpSize >> 1); offset; offset >>= 1) {
            val += __shfl_down_sync(0xFFFFFFFF, val, offset);
        }
    }
    return val;
}

__global__ void reduce_v4(float* g_idata, float* g_odata, int n) {
    float sum = 0.0f;
    for (int i = (blockIdx.x * blockDim.x) + threadIdx.x; i < n; i += (gridDim.x * blockDim.x)) {
        sum += g_idata[i];
    }
    sum = block_reduce_sum(sum);
    if (threadIdx.x == 0) {
        g_odata[blockIdx.x] = sum;
    }
}

float reduce_cpu(const std::vector<float>& data) {
    float sum = 0.0f;
    for (float val : data) {
        sum += val;
    }
    return sum;
}

int main() {
    std::vector<float> h_data(N);
    for (int i = 0; i < N; ++i) {
        h_data[i] = 1.0f;
    }

    // CPU 计时开始
    auto start_cpu = std::chrono::high_resolution_clock::now();
    float out_cpu = reduce_cpu(h_data);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpu_time = end_cpu - start_cpu;
    // CPU 计时结束

    float* d_data;
    float* d_out;
    float* d_final_out;
    float out_cu;

    cudaMalloc(&d_data, N * sizeof(float));
    cudaMalloc(&d_out, block_num * sizeof(float));
    cudaMalloc(&d_final_out, sizeof(float));
    cudaMemcpy(d_data, h_data.data(), N * sizeof(float), cudaMemcpyHostToDevice);

    for (int i = 0; i < 10; ++i) {
        reduce_v4<<<block_num, thread_num>>>(d_data, d_out, N);
        reduce_v4<<<1, block_num>>>(d_out, d_final_out, block_num);
    }

    // GPU 计时开始 (CUDA Events)
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    reduce_v4<<<block_num, thread_num>>>(d_data, d_out, N);
    reduce_v4<<<1, block_num>>>(d_out, d_final_out, block_num);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    // GPU 计时结束

    cudaMemcpy(&out_cu, d_final_out, sizeof(float), cudaMemcpyDeviceToHost);
    
    if (fabs(out_cpu - out_cu) < 1e-5) {
        std::cout << "Result verified successfully!" << std::endl;
    } else {
        std::cout << "Result verification failed!" << std::endl;
    }
    std::cout << "CPU result: " << out_cpu << std::endl;
    std::cout << "GPU result: " << out_cu << std::endl;
    std::cout << "CPU time: " << cpu_time.count() << " ms" << std::endl;
    std::cout << "GPU time: " << milliseconds << " ms" << std::endl;
    std::cout << "Speedup: " << (cpu_time.count() / milliseconds) << "x" << std::endl;

    cudaFree(d_data);
    cudaFree(d_out);
    cudaFree(d_final_out);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}